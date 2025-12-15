"""
Evaluate PPO Agent and Compare with LP Baseline
=================================================

This script evaluates trained PPO models and compares performance
with the Linear Programming (LP) baseline from vm_resource_planner.py.

Usage:
    python eval_ppo.py --scenario overload
    python eval_ppo.py --scenario cost
    python eval_ppo.py --scenario both

Outputs:
    - PPO schedules: forecast_result/ppo_schedule_test_<scenario>.csv
    - Comparison report: forecast_result/ppo_vs_lp_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.config import SCENARIO_OVERLOAD, SCENARIO_COST, HOST_SPEC
from rl.environment import make_env
from rl.utils import (
    load_data,
    load_vm_catalog,
    compute_resource_requirements,
    compute_vm_resources,
    format_allocation,
    evaluate_episode,
)
from rl.reward import compute_switching_cost


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "rl_models"
RESULTS_DIR = PROJECT_ROOT / "forecast_result"
LP_SCHEDULE_PATH = RESULTS_DIR / "vm_schedule.csv"

RESULTS_DIR.mkdir(exist_ok=True)


def load_ppo_model(
    scenario: str,
    episode_length: Optional[int] = None,
) -> tuple[PPO, Optional[VecNormalize]]:
    """
    Load trained PPO model and VecNormalize statistics.
    
    Args:
        scenario: Scenario name
        episode_length: Optional episode length override
        
    Returns:
        Tuple of (model, vec_normalize)
    """
    model_path = MODELS_DIR / f"ppo_{scenario}.zip"
    vec_normalize_path = MODELS_DIR / f"ppo_{scenario}_vecnormalize.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please run 'python train_ppo.py --scenario {scenario}' first."
        )
    
    # Load model
    model = PPO.load(str(model_path))
    
    # Load VecNormalize if available
    vec_normalize = None
    if vec_normalize_path.exists():
        # Create a dummy env to attach VecNormalize
        # NOTE: Do NOT pass horizon - let env use default to match trained model
        def make_dummy_env():
            env_kwargs = {}
            if episode_length is not None:
                env_kwargs["episode_length"] = episode_length
            return make_env(scenario=scenario, is_training=False, **env_kwargs)
        
        env = DummyVecEnv([make_dummy_env])
        try:
            vec_normalize = VecNormalize.load(str(vec_normalize_path), env)
            vec_normalize.training = False
            vec_normalize.norm_reward = False
        except AssertionError as e:
            print(f"Warning: VecNormalize shape mismatch, skipping normalization. Details: {e}")
            vec_normalize = None
    
    return model, vec_normalize


def rollout_ppo(
    model: PPO,
    scenario: str,
    vec_normalize: Optional[VecNormalize] = None,
    episode_length: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run PPO model on test data and collect results.
    
    Args:
        model: Trained PPO model
        scenario: Scenario name
        vec_normalize: VecNormalize for observation normalization
        episode_length: Number of steps to evaluate (None = use default from config)
        
    Returns:
        DataFrame with PPO allocations and metrics
    """
    print(f"\nRunning PPO rollout for scenario: {scenario}")
    
    # Load test data
    _, test_df = load_data()
    vm_catalog = load_vm_catalog()
    vm_types = list(vm_catalog.keys())
    
    # Create test environment with optional custom episode_length
    # NOTE: Do NOT pass horizon - let env use default to match trained model
    env_kwargs = {}
    if episode_length is not None:
        env_kwargs["episode_length"] = episode_length
    
    env = make_env(scenario=scenario, is_training=False, **env_kwargs)
    
    # Wrap with VecNormalize if available
    if vec_normalize is not None:
        # Use the loaded VecNormalize (VecEnv API)
        # VecNormalize.reset() (VecEnv) → obs (không trả về info)
        vec_env = vec_normalize
        obs = vec_env.reset()
    else:
        # Just use the raw Gymnasium environment
        # Gymnasium Env.reset() → (obs, info)
        obs, _ = env.reset()
        vec_env = None
    
    records = []
    prev_allocation = {}
    done = False
    step = 0
    
    # Get episode length from environment
    episode_length = env.episode_length
    
    while step < min(episode_length, len(test_df)):
        # Get action from model
        if vec_env is not None:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_flags, info = vec_env.step(action)
            done = done_flags[0] if hasattr(done_flags, '__len__') else done_flags
            # VecEnv returns action with shape (n_envs, n_actions), extract first env
            if len(action.shape) > 1:
                action = action[0]
        else:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Flatten action if needed (in case of nested arrays)
        action = np.asarray(action).flatten()
        
        # Extract allocation from action
        allocation = {}
        for i, vm_type in enumerate(vm_types):
            count = int(action[i])
            if count > 0:
                allocation[vm_type] = count
        
        # Get current data row
        row = test_df.iloc[step]
        cpu_req, mem_req, cpu_over, mem_over = compute_resource_requirements(
            row, HOST_SPEC
        )
        cpu_alloc_vm, mem_alloc_vm, vm_cost = compute_vm_resources(allocation, vm_catalog)
        
        # Host thresholds (what host can handle before needing VMs)
        cpu_threshold = HOST_SPEC["total_cpu_cores"] * HOST_SPEC["cpu_threshold_pct"] / 100.0
        mem_threshold = HOST_SPEC["total_memory_gb"] * HOST_SPEC["memory_threshold_pct"] / 100.0
        
        # Total available = Host threshold + VM resources
        cpu_alloc_total = cpu_threshold + cpu_alloc_vm
        mem_alloc_total = mem_threshold + mem_alloc_vm
        
        # Compute switching cost
        switch_cost = compute_switching_cost(prev_allocation, allocation, vm_catalog)
        
        # Compute utilization (based on VM allocation only, since host is always used)
        # If no VMs, utilization is 0% (host handles everything)
        if cpu_alloc_vm > 0:
            cpu_util = (max(0, cpu_req - cpu_threshold) / cpu_alloc_vm * 100)
        else:
            cpu_util = 0 if cpu_req <= cpu_threshold else 100  # overflow without VMs
        
        if mem_alloc_vm > 0:
            mem_util = (max(0, mem_req - mem_threshold) / mem_alloc_vm * 100)
        else:
            mem_util = 0 if mem_req <= mem_threshold else 100  # overflow without VMs
        
        # Check SLA violation: total available must meet total demand
        sla_violated = (cpu_alloc_total < cpu_req) or (mem_alloc_total < mem_req)
        
        # Create record (before host-only postprocess)
        record = {
            "timestamp": row.get("datetime", step),
            "allocation": format_allocation(allocation),
            "vm_cost_per_hour": vm_cost,
            "switching_cost": switch_cost,
            "total_cost_per_hour": vm_cost + switch_cost,
            "cpu_allocated_cores": cpu_alloc_total,  # Host + VMs
            "mem_allocated_gb": mem_alloc_total,      # Host + VMs
            "cpu_vm_only": cpu_alloc_vm,              # VMs only
            "mem_vm_only": mem_alloc_vm,              # VMs only
            "cpu_required_cores": cpu_req,
            "mem_required_gb": mem_req,
            "cpu_overflow_cores": cpu_over,           # Demand exceeding host threshold
            "mem_overflow_gb": mem_over,              # Demand exceeding host threshold
            "cpu_utilization_pct": min(100, cpu_util),
            "mem_utilization_pct": min(100, mem_util),
            "sla_violation_flag": int(sla_violated),
        }
        
        # Host-only postprocess: if no overflow, force Host only (align with LP)
        if cpu_over <= 0 and mem_over <= 0:
            record["allocation"] = "Host only"
            record["vm_cost_per_hour"] = 0.0
            record["switching_cost"] = 0.0
            record["total_cost_per_hour"] = 0.0
            record["cpu_allocated_cores"] = cpu_threshold  # host threshold only
            record["mem_allocated_gb"] = mem_threshold
            record["cpu_vm_only"] = 0.0
            record["mem_vm_only"] = 0.0
        records.append(record)
        
        prev_allocation = allocation.copy()
        step += 1
        
        if done:
            break
    
    # Clean up
    if vec_env is not None:
        vec_env.close()
    else:
        env.close()
    
    df = pd.DataFrame(records)
    
    # Resample to 30-minute buckets to match LP outputs
    if df.empty:
        # Return empty DataFrames if no records
        return df, pd.DataFrame()
    
    # Ensure timestamp column exists and is datetime
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="30S")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    
    # Check which columns exist before resampling
    required_cols = [
        "vm_cost_per_hour", "switching_cost", "total_cost_per_hour",
        "cpu_required_cores", "mem_required_gb",
        "cpu_overflow_cores", "mem_overflow_gb",
        "cpu_vm_only", "mem_vm_only",
        "sla_violation_flag"
    ]
    
    # Verify all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in DataFrame: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
    
    # Only aggregate columns that exist
    agg_dict = {}
    for col in required_cols:
        if col in df.columns:
            if col == "sla_violation_flag":
                agg_dict[col] = ["sum", "mean"]
            elif col in ["vm_cost_per_hour", "switching_cost", "total_cost_per_hour"]:
                agg_dict[col] = "sum"
            else:
                agg_dict[col] = "max"
    
    if not agg_dict:
        # No columns to aggregate, return empty bucket
        print("Warning: No columns available for aggregation")
        bucket = pd.DataFrame()
    else:
        bucket = df.resample("30T").agg(agg_dict)
        
        # Flatten MultiIndex columns if needed
        if isinstance(bucket.columns, pd.MultiIndex):
            bucket.columns = [
                "_".join(col).strip("_") if isinstance(col, tuple) else col 
                for col in bucket.columns
            ]
    
    # Reset index for saving
    df = df.reset_index()
    if not bucket.empty:
        bucket = bucket.reset_index()
    
    return df, bucket


def load_lp_baseline() -> pd.DataFrame:
    """
    Load LP baseline schedule from vm_schedule.csv.
    
    Returns:
        DataFrame with LP allocations
    """
    if not LP_SCHEDULE_PATH.exists():
        print(f"Warning: LP schedule not found at {LP_SCHEDULE_PATH}")
        print("Run 'python vm_resource_planner.py' first to generate baseline.")
        return pd.DataFrame()
    
    return pd.read_csv(LP_SCHEDULE_PATH)


def compare_with_lp(
    ppo_df: pd.DataFrame,
    ppo_bucket_df: pd.DataFrame | None,
    lp_df: pd.DataFrame,
    scenario: str,
) -> Dict[str, Any]:
    """
    Compare PPO results with LP baseline.
    
    Args:
        ppo_df: PPO schedule DataFrame
        lp_df: LP schedule DataFrame
        scenario: Scenario name
        
    Returns:
        Comparison metrics dictionary
    """
    comparison = {
        "scenario": scenario,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # PPO metrics (per-step)
    if not ppo_df.empty:
        comparison["ppo"] = {
            "total_vm_cost": float(ppo_df["vm_cost_per_hour"].sum()),
            "total_switching_cost": float(ppo_df["switching_cost"].sum()),
            "total_cost": float(ppo_df["total_cost_per_hour"].sum()),
            "sla_violations": int(ppo_df["sla_violation_flag"].sum()),
            "sla_violation_rate": float(ppo_df["sla_violation_flag"].mean()),
            "mean_cpu_utilization": float(ppo_df["cpu_utilization_pct"].mean()),
            "mean_mem_utilization": float(ppo_df["mem_utilization_pct"].mean()),
            "n_steps": len(ppo_df),
        }
    # PPO bucket metrics (30min)
    if ppo_bucket_df is not None and not ppo_bucket_df.empty:
        # Columns after flatten: vm_cost_per_hour, switching_cost, total_cost_per_hour, sla_violation_flag_sum, sla_violation_flag_mean
        comparison["ppo_30min"] = {
            "total_vm_cost": float(ppo_bucket_df["vm_cost_per_hour"].sum() if "vm_cost_per_hour" in ppo_bucket_df.columns else 0.0),
            "total_switching_cost": float(ppo_bucket_df["switching_cost"].sum() if "switching_cost" in ppo_bucket_df.columns else 0.0),
            "total_cost": float(ppo_bucket_df["total_cost_per_hour"].sum() if "total_cost_per_hour" in ppo_bucket_df.columns else 0.0),
            "sla_violations": int(ppo_bucket_df["sla_violation_flag_sum"].sum() if "sla_violation_flag_sum" in ppo_bucket_df.columns else 0),
            "sla_violation_rate": float(ppo_bucket_df["sla_violation_flag_mean"].mean() if "sla_violation_flag_mean" in ppo_bucket_df.columns else 0.0),
            "n_buckets": len(ppo_bucket_df),
        }
    
    # LP metrics (from min_cost scenario)
    if not lp_df.empty:
        lp_cost_col = "min_cost_cost_per_hour"
        if lp_cost_col in lp_df.columns:
            comparison["lp"] = {
                "total_vm_cost": float(lp_df[lp_cost_col].sum()),
                "total_switching_cost": 0.0,  # LP doesn't consider switching
                "total_cost": float(lp_df[lp_cost_col].sum()),
                "n_steps": len(lp_df),
            }
            
            # Calculate improvement
            if comparison.get("ppo") and comparison.get("lp"):
                ppo_cost = comparison["ppo"]["total_cost"]
                lp_cost = comparison["lp"]["total_cost"]
                
                if lp_cost > 0:
                    cost_diff = lp_cost - ppo_cost
                    cost_pct = (cost_diff / lp_cost) * 100
                    comparison["improvement"] = {
                        "cost_difference": float(cost_diff),
                        "cost_improvement_pct": float(cost_pct),
                        "ppo_better": cost_diff > 0,
                    }
    
    return comparison


def evaluate_scenario(
    scenario: str,
    episode_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline for a scenario.
    
    Args:
        scenario: Scenario name
        episode_length: Number of steps to evaluate (None = use default 480)
        
    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Evaluating scenario: {scenario.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load model
        model, vec_normalize = load_ppo_model(scenario, episode_length=episode_length)
        print(f"Loaded model from {MODELS_DIR / f'ppo_{scenario}.zip'}")
        
        # Run PPO rollout
        ppo_df, ppo_bucket = rollout_ppo(
            model,
            scenario,
            vec_normalize,
            episode_length=episode_length,
        )
        
        # Save PPO schedule (per-step) and bucketed 30min
        ppo_schedule_path = RESULTS_DIR / f"ppo_schedule_test_{scenario}.csv"
        ppo_bucket_path = RESULTS_DIR / f"ppo_schedule_30min_{scenario}.csv"
        ppo_df.to_csv(ppo_schedule_path, index=False)
        ppo_bucket.to_csv(ppo_bucket_path, index=False)
        print(f"PPO schedule saved to {ppo_schedule_path}")
        print(f"PPO 30min bucket saved to {ppo_bucket_path}")
        
        # Load LP baseline
        lp_df = load_lp_baseline()
        
        # Compare (pass bucketed df too)
        comparison = compare_with_lp(ppo_df, ppo_bucket, lp_df, scenario)
        
        return comparison
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"scenario": scenario, "error": str(e)}


def print_comparison(comparison: Dict[str, Any]):
    """Print comparison results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {comparison.get('scenario', 'Unknown').upper()}")
    print(f"{'='*60}")
    
    if "error" in comparison:
        print(f"Error: {comparison['error']}")
        return
    
    if "ppo" in comparison:
        ppo = comparison["ppo"]
        print("\nPPO Performance:")
        print(f"  Total VM Cost: ${ppo['total_vm_cost']:.4f}")
        print(f"  Total Switching Cost: ${ppo['total_switching_cost']:.4f}")
        print(f"  Total Cost: ${ppo['total_cost']:.4f}")
        print(f"  SLA Violations: {ppo['sla_violations']} ({ppo['sla_violation_rate']*100:.1f}%)")
        print(f"  Mean CPU Utilization: {ppo['mean_cpu_utilization']:.1f}%")
        print(f"  Mean Memory Utilization: {ppo['mean_mem_utilization']:.1f}%")
    
    if "lp" in comparison:
        lp = comparison["lp"]
        print("\nLP Baseline Performance:")
        print(f"  Total VM Cost: ${lp['total_vm_cost']:.4f}")
        print(f"  Total Cost: ${lp['total_cost']:.4f}")
        print(f"  (LP does not consider switching cost)")
    
    if "improvement" in comparison:
        imp = comparison["improvement"]
        print("\nComparison:")
        direction = "lower" if imp["ppo_better"] else "higher"
        print(f"  PPO cost is ${abs(imp['cost_difference']):.4f} {direction} than LP")
        print(f"  Cost difference: {imp['cost_improvement_pct']:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO and compare with LP baseline"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["overload", "cost", "both"],
        default="both",
        help="Scenario to evaluate",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Number of steps to evaluate (default: env config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for comparison results",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("PPO Evaluation for VM Allocation")
    print("="*60)
    
    scenarios = []
    if args.scenario == "both":
        scenarios = [SCENARIO_OVERLOAD, SCENARIO_COST]
    else:
        scenarios = [args.scenario]
    
    all_comparisons = []
    
    for scenario in scenarios:
        comparison = evaluate_scenario(
            scenario,
            episode_length=args.episode_length,
        )
        all_comparisons.append(comparison)
        print_comparison(comparison)
    
    # Save comparison results
    output_path = args.output or str(RESULTS_DIR / "ppo_vs_lp_comparison.json")
    with open(output_path, "w") as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"\nComparison results saved to {output_path}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()


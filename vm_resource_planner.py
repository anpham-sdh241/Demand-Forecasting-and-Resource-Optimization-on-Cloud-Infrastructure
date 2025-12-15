"""
VM Resource Planning Pipeline
=============================

This script connects the forecasting layer (Hybrid Prophet + LSTM)
with resource planning for VM allocation.  It implements three phases:

1. Forecast → Resource Requirements
2. Resource Requirements → VM Allocation (Linear Programming)
3. VM Scheduling & Optimization

Outputs:
- Peak resource requirements
- VM allocation plans for two scenarios:
    a) Minimize Resource Overload
    b) Optimize Operational Cost
- Time-bucketed scheduling recommendations
- JSON report + optional CSV exports
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import linprog

from model_utils import get_latest_model, load_model, save_results
from rl.reward import compute_switching_cost, compute_vm_cost

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
VM_TYPES_FILE = PROJECT_ROOT / "VMs_type.json"
NORMALIZATION_STATS_FILE = PROJECT_ROOT / "processed_data" / "normalization_stats.json"
RESULTS_DIR = PROJECT_ROOT / "forecast_result"
RESULTS_DIR.mkdir(exist_ok=True)

TARGETS = ["memory_usage_pct", "cpu_total_usage", "system_load"]
SUPPORTED_MODELS = {
    "hybrid_prophet_lstm": "hybrid",
    "arimax": "stats",
    "random_forest": "ml",
    "svr": "ml",
}
DEFAULT_MODEL_NAME = "hybrid_prophet_lstm"
SERIES_FREQ = "30S"
START_TIMESTAMP = pd.Timestamp("2024-01-01 00:00:00")
STEP_HOURS = pd.to_timedelta(SERIES_FREQ).total_seconds() / 3600.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Host machine specification (can be adjusted)
HOST_SPEC = {
    "total_cpu_cores": 1,
    "total_memory_gb": 4,
    "cpu_threshold_pct": 70,     # begin offloading after 70% CPU usage
    "memory_threshold_pct": 75,  # begin offloading after 75% memory usage
}

# Scheduling configuration
SCHEDULING_BUCKET = "30T"  # aggregate every 30 minutes


# -----------------------------------------------------------------------------
# Helper classes / functions reused from inference
# -----------------------------------------------------------------------------

def load_series(target: str) -> tuple[pd.Series, pd.Series]:
    target_dir = DATA_DIR / target
    y_train = pd.read_csv(target_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(target_dir / "y_test.csv").squeeze()
    return y_train, y_test


class LSTMResidualModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def forecast_residuals(model, seed_sequence, n_steps, window_size):
    model.eval()
    seq = seed_sequence.copy().tolist()
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            window = torch.tensor(seq[-window_size:], dtype=torch.float32).view(
                1, window_size, 1
            ).to(DEVICE)
            pred = model(window).cpu().item()
            preds.append(pred)
            seq.append(pred)
    return np.array(preds)


def inverse_scale(values, mean, std):
    if std == 0:
        return np.full_like(values, mean)
    return values * std + mean


def load_normalization_stats() -> Dict[str, Dict[str, float]]:
    """Load normalization statistics for inverse transformation."""
    if not NORMALIZATION_STATS_FILE.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {NORMALIZATION_STATS_FILE}\n"
            "Please run ETL pipeline to generate this file."
        )
    with open(NORMALIZATION_STATS_FILE, "r") as f:
        stats = json.load(f)
    return stats["targets"]


def inverse_transform_forecasts(
    forecast_df: pd.DataFrame,
    norm_stats: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Convert normalized forecast values back to original scale.
    
    Args:
        forecast_df: DataFrame with normalized forecast values
        norm_stats: Dictionary with mean/std for each target
        
    Returns:
        DataFrame with values in original scale
    """
    df = forecast_df.copy()
    
    for target in TARGETS:
        if target not in df.columns:
            continue
        if target not in norm_stats:
            print(f"Warning: No normalization stats for {target}, skipping inverse transform")
            continue
            
        mean = norm_stats[target]["mean"]
        std = norm_stats[target]["std"]
        
        # Inverse transform: x_original = x_normalized * std + mean
        df[target] = df[target] * std + mean
        
    return df


# -----------------------------------------------------------------------------
# Forecast loading
# -----------------------------------------------------------------------------

def load_latest_models(model_name: str) -> Dict[str, Path]:
    model_paths: Dict[str, Path] = {}
    for target in TARGETS:
        model_paths[target] = Path(
            get_latest_model(
                models_dir=str(MODELS_DIR),
                model_name=model_name,
                target=target,
            )
        )
    return model_paths


def load_ground_truth_df() -> pd.DataFrame:
    """
    Build a DataFrame that mimics a streaming ground-truth feed using y_test of all targets.

    Returns:
        DataFrame with columns: timestamp, memory_usage_pct, cpu_total_usage, system_load
    """
    records: Dict[str, np.ndarray] = {}
    timestamps = None

    for target in TARGETS:
        _y_train, y_test = load_series(target)
        records[target] = y_test.values
        if timestamps is None:
            time_index = pd.date_range(
                start=START_TIMESTAMP, periods=len(y_test), freq=SERIES_FREQ
            )
            timestamps = time_index

    if timestamps is None:
        raise RuntimeError("No ground truth available.")

    df = pd.DataFrame({"timestamp": timestamps})
    for target in TARGETS:
        df[target] = records[target]
    return df


def generate_forecasts(
    model_paths: Dict[str, Path],
    model_name: str,
    apply_inverse_transform: bool = True,
) -> pd.DataFrame:
    """
    Run inference and build a forecast DataFrame for the selected model.
    
    Args:
        model_paths: Dictionary mapping target names to model file paths
        model_name: Name of the model to use
        apply_inverse_transform: If True, convert normalized predictions back to original scale
        
    Returns:
        DataFrame with forecast values (in original scale if apply_inverse_transform=True)
    """
    model_type = SUPPORTED_MODELS.get(model_name)
    if model_type is None:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_type == "hybrid":
        forecast_df = _generate_forecasts_hybrid(model_paths)
    elif model_type == "stats":
        forecast_df = _generate_forecasts_stats(model_paths, model_name)
    else:
        forecast_df = _generate_forecasts_ml(model_paths, model_name)
    
    # Apply inverse transformation to convert normalized values back to original scale
    if apply_inverse_transform:
        norm_stats = load_normalization_stats()
        forecast_df = inverse_transform_forecasts(forecast_df, norm_stats)
        print("✓ Applied inverse transformation to forecasts (original scale)")
    
    return forecast_df


def _generate_forecasts_hybrid(model_paths: Dict[str, Path]) -> pd.DataFrame:
    forecast_records: Dict[str, np.ndarray] = {}
    timestamps = None

    for target in TARGETS:
        y_train, y_test = load_series(target)
        n_train = len(y_train)
        n_test = len(y_test)
        total_len = n_train + n_test

        time_index = pd.date_range(
            start=START_TIMESTAMP, periods=total_len, freq=SERIES_FREQ
        )
        test_index = time_index[-n_test:]

        model_bundle, metadata = load_model(str(model_paths[target]))
        prophet_model = model_bundle["prophet"]
        lstm_state = model_bundle["lstm_state_dict"]
        residual_mean = model_bundle["residual_mean"]
        residual_std = model_bundle["residual_std"] if model_bundle["residual_std"] > 0 else 1e-6
        window_size = model_bundle.get("window_size") or model_bundle.get(
            "config", {}
        ).get("lstm", {}).get("window_size", 96)
        freq = model_bundle.get("freq", SERIES_FREQ)
        lstm_cfg = model_bundle.get("config", {}).get("lstm", {})

        lstm_model = LSTMResidualModel(
            hidden_size=lstm_cfg.get("hidden_size", 64),
            num_layers=lstm_cfg.get("num_layers", 2),
            dropout=lstm_cfg.get("dropout", 0.1),
        ).to(DEVICE)
        lstm_model.load_state_dict(lstm_state)
        lstm_model.eval()

        future_df = prophet_model.make_future_dataframe(periods=n_test, freq=freq)
        forecast_df = prophet_model.predict(future_df)
        prophet_preds = forecast_df["yhat"].values
        prophet_train_pred = prophet_preds[:n_train]
        prophet_test_pred = prophet_preds[-n_test:]

        train_residuals = y_train.values - prophet_train_pred
        residuals_scaled = (train_residuals - residual_mean) / residual_std

        if len(residuals_scaled) < window_size:
            raise ValueError(
                f"Not enough residual history ({len(residuals_scaled)}) for window_size={window_size}"
            )

        seed_sequence = residuals_scaled[-window_size:]
        residual_test_scaled = forecast_residuals(
            lstm_model, seed_sequence, n_test, window_size
        )
        residual_test_pred = residual_test_scaled * residual_std + residual_mean
        hybrid_test_pred = prophet_test_pred + residual_test_pred

        forecast_records[target] = hybrid_test_pred
        timestamps = test_index

    if timestamps is None:
        raise RuntimeError("No predictions generated.")

    forecast_df = pd.DataFrame({"timestamp": timestamps})
    for target, values in forecast_records.items():
        forecast_df[target] = values
    return forecast_df


def _generate_forecasts_ml(
    model_paths: Dict[str, Path], model_name: str
) -> pd.DataFrame:
    forecast_records: Dict[str, np.ndarray] = {}
    timestamps = None

    for target in TARGETS:
        y_train, y_test = load_series(target)
        X_test = pd.read_csv(DATA_DIR / target / "X_test.csv")
        model, metadata = load_model(str(model_paths[target]))

        start_idx = len(y_train)
        total_len = len(y_train) + len(y_test)
        time_index = pd.date_range(
            start=START_TIMESTAMP, periods=total_len, freq=SERIES_FREQ
        )
        test_index = time_index[start_idx:]

        if model_name == "arimax":
            preds = model.forecast(steps=len(y_test), exog=X_test)
        else:
            preds = model.predict(X_test)

        preds = np.asarray(preds).flatten()
        forecast_records[target] = preds
        timestamps = test_index

    if timestamps is None:
        raise RuntimeError("No predictions generated.")

    forecast_df = pd.DataFrame({"timestamp": timestamps})
    for target, values in forecast_records.items():
        forecast_df[target] = values
    return forecast_df


def _generate_forecasts_stats(
    model_paths: Dict[str, Path], model_name: str
) -> pd.DataFrame:
    return _generate_forecasts_ml(model_paths, model_name)


# -----------------------------------------------------------------------------
# Phase 1: Forecast → Resource Requirements
# -----------------------------------------------------------------------------

def convert_forecasts_to_requirements(
    forecast_df: pd.DataFrame, host_spec: Dict[str, float]
) -> pd.DataFrame:

    df = forecast_df.copy()

    # Clip negative values to 0
    mem_pct = df["memory_usage_pct"].clip(lower=0)
    
    # cpu_total_usage = "rate of CPU seconds/second" = số cores đang sử dụng
    # Đây là giá trị ABSOLUTE (cores), KHÔNG PHẢI percentage
    cpu_cores_used = df["cpu_total_usage"].clip(lower=0)
    
    # system_load = load average = số processes đang chờ/chạy trên CPU
    system_load = df["system_load"].clip(lower=0)

    # =========================================================================
    # MEMORY CALCULATIONS
    # =========================================================================
    # memory_required_gb: Chuyển % thành GB thực tế
    mem_required_gb = host_spec["total_memory_gb"] * mem_pct / 100.0
    
    # memory_threshold_gb: Ngưỡng an toàn (VD: 75% của tổng RAM)
    mem_threshold_gb = (
        host_spec["total_memory_gb"] * host_spec["memory_threshold_pct"] / 100.0
    )
    
    # memory_overflow_gb: Phần RAM vượt ngưỡng cần offload
    mem_overflow_gb = np.maximum(0.0, mem_required_gb - mem_threshold_gb)

    # =========================================================================
    # CPU CALCULATIONS
    # =========================================================================
    # cpu_required_cores: Lấy max của CPU usage và load average
    # - cpu_cores_used: cores đang consume CPU time
    # - system_load: processes đang cạnh tranh CPU
    cpu_required_cores = np.maximum(cpu_cores_used, system_load)
    
    # cpu_threshold_cores: Ngưỡng an toàn (VD: 70% của tổng cores)
    cpu_threshold_cores = (
        host_spec["total_cpu_cores"] * host_spec["cpu_threshold_pct"] / 100.0
    )
    
    # cpu_overflow_cores: Phần CPU vượt ngưỡng cần offload sang VM
    cpu_overflow_cores = np.maximum(0.0, cpu_required_cores - cpu_threshold_cores)

    df["memory_required_gb"] = mem_required_gb
    df["memory_overflow_gb"] = mem_overflow_gb
    df["cpu_required_cores"] = cpu_required_cores
    df["cpu_overflow_cores"] = cpu_overflow_cores

    return df


# -----------------------------------------------------------------------------
# Phase 2: Resource Requirements → VM Allocation (Linear Programming)
# -----------------------------------------------------------------------------

def load_vm_catalog(vm_file: Path) -> List[Dict[str, Any]]:
    with open(vm_file, "r") as f:
        vm_data = json.load(f)

    catalog = []
    for name, spec in vm_data.items():
        catalog.append(
            {
                "name": name,
                "vcpus": spec["vcpus"],
                "memory_gb": spec["memory_gb"],
                "cost_per_hour": spec["cost_per_hour"],
                "switching_cost": spec.get("switching_cost", 0.0),  # Include switching cost
            }
        )
    catalog.sort(key=lambda item: item["cost_per_hour"])
    return catalog


def solve_vm_allocation(
    cpu_req: float,
    mem_req: float,
    vm_catalog: List[Dict[str, Any]],
    objective: str = "capacity",
    cpu_weight: float = 1.0,
    mem_weight: float = 0.25,
) -> Dict[str, Any]:
    """
    Solve VM allocation using Linear Programming.
    
    Objectives:
    - "cost": Minimize total cost (prefer cheapest VMs)
    - "capacity": Minimize cost per capacity unit (prefer VMs with better cost-effectiveness)
    """
    cpu_req = max(0.0, float(cpu_req))
    mem_req = max(0.0, float(mem_req))

    if cpu_req <= 0 and mem_req <= 0:
        return {
            "allocation": {},
            "total_vms": 0,
            "total_cpu": 0.0,
            "total_memory": 0.0,
            "total_cost": 0.0,
        }

    num_vm = len(vm_catalog)
    if objective == "cost":
        # Minimize pure cost → prefers cheapest VMs
        c = [vm["cost_per_hour"] for vm in vm_catalog]
    else:
        # Minimize cost per capacity → prefers VMs with better $/capacity ratio
        # Lower ratio = better value for capacity
        # This makes larger VMs more attractive when they're cost-effective
        c = []
        for vm in vm_catalog:
            capacity = cpu_weight * vm["vcpus"] + mem_weight * vm["memory_gb"]
            cost_per_capacity = vm["cost_per_hour"] / capacity if capacity > 0 else float('inf')
            c.append(cost_per_capacity)

    A_ub = [
        [-vm["vcpus"] for vm in vm_catalog],
        [-vm["memory_gb"] for vm in vm_catalog],
    ]
    b_ub = [-cpu_req, -mem_req]
    bounds = [(0, None) for _ in range(num_vm)]

    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Linear program failed: {result.message}")

    counts = np.ceil(result.x - 1e-9).astype(int)
    allocation = {}
    total_cpu = 0.0
    total_mem = 0.0
    total_cost = 0.0

    for vm, count in zip(vm_catalog, counts):
        if count <= 0:
            continue
        allocation[vm["name"]] = int(count)
        total_cpu += count * vm["vcpus"]
        total_mem += count * vm["memory_gb"]
        total_cost += count * vm["cost_per_hour"]

    return {
        "allocation": allocation,
        "total_vms": int(sum(allocation.values())),
        "total_cpu": float(total_cpu),
        "total_memory": float(total_mem),
        "total_cost": float(total_cost),
    }


def summarize_peak_plans(
    requirements_df: pd.DataFrame,
    vm_catalog: List[Dict[str, Any]],
) -> Dict[str, Any]:
    peak_cpu = float(requirements_df["cpu_overflow_cores"].max())
    peak_mem = float(requirements_df["memory_overflow_gb"].max())

    overload_plan = solve_vm_allocation(
        peak_cpu,
        peak_mem,
        vm_catalog,
        objective="capacity",
        cpu_weight=1.0,
        mem_weight=0.5,
    )
    cost_plan = solve_vm_allocation(
        peak_cpu,
        peak_mem,
        vm_catalog,
        objective="cost",
    )

    return {
        "peak_cpu_overflow": peak_cpu,
        "peak_memory_overflow": peak_mem,
        "minimize_overload_plan": overload_plan,
        "minimize_cost_plan": cost_plan,
    }


# -----------------------------------------------------------------------------
# Phase 3: VM Scheduling & Optimization
# -----------------------------------------------------------------------------

def format_allocation(allocation: Dict[str, int]) -> str:
    if not allocation:
        return "No VMs"
    return ", ".join(f"{vm}×{count}" for vm, count in allocation.items())


def build_schedule(
    requirements_df: pd.DataFrame,
    vm_catalog: List[Dict[str, Any]],
    bucket: str = SCHEDULING_BUCKET,
) -> pd.DataFrame:
    df = requirements_df.set_index("timestamp")
    bucketed = (
        df[["cpu_overflow_cores", "memory_overflow_gb"]]
        .resample(bucket)
        .max()
        .fillna(0.0)
    )
    records = []

    for ts, row in bucketed.iterrows():
        cpu_req = float(row["cpu_overflow_cores"])
        mem_req = float(row["memory_overflow_gb"])

        if cpu_req <= 0 and mem_req <= 0:
            records.append(
                {
                    "timestamp": ts,
                    "cpu_overflow_cores": 0.0,
                    "memory_overflow_gb": 0.0,
                    "min_overload_plan": "Host only",
                    "min_overload_cost_per_hour": 0.0,
                    "min_cost_plan": "Host only",
                    "min_cost_cost_per_hour": 0.0,
                }
            )
            continue

        overload_plan = solve_vm_allocation(
            cpu_req,
            mem_req,
            vm_catalog,
            objective="capacity",
            cpu_weight=1.0,
            mem_weight=0.5,
        )
        cost_plan = solve_vm_allocation(
            cpu_req,
            mem_req,
            vm_catalog,
            objective="cost",
        )

        records.append(
            {
                "timestamp": ts,
                "cpu_overflow_cores": cpu_req,
                "memory_overflow_gb": mem_req,
                "min_overload_plan": format_allocation(overload_plan["allocation"]),
                "min_overload_cost_per_hour": overload_plan["total_cost"],
                "min_cost_plan": format_allocation(cost_plan["allocation"]),
                "min_cost_cost_per_hour": cost_plan["total_cost"],
            }
        )

    schedule_df = pd.DataFrame(records)
    return schedule_df


def build_reactive_schedule_for_scenario(
    requirements_df: pd.DataFrame,
    vm_catalog: List[Dict[str, Any]],
    scenario: str = "cost",
) -> pd.DataFrame:
    """
    Solve LP per timestamp for a specific scenario.
    
    Args:
        requirements_df: DataFrame with cpu_overflow_cores, memory_overflow_gb
        vm_catalog: List of VM specifications
        scenario: "cost" or "overload"
        
    Returns:
        DataFrame with per-step allocation and metrics
    """
    records = []
    prev_alloc: Dict[str, int] = {}
    vm_catalog_map = {vm["name"]: vm for vm in vm_catalog}
    
    objective = "cost" if scenario == "cost" else "capacity"

    for _, row in requirements_df.iterrows():
        cpu_req = float(row["cpu_overflow_cores"])
        mem_req = float(row["memory_overflow_gb"])

        if cpu_req <= 0 and mem_req <= 0:
            records.append(
                {
                    "timestamp": row["timestamp"],
                    "cpu_overflow_cores": cpu_req,
                    "memory_overflow_gb": mem_req,
                    "allocation": "Host only",
                    "vm_total_count": 0,
                    "vm_cost_per_hour": 0.0,
                    "vm_cost_per_step": 0.0,
                    "switching_cost": 0.0,
                    "total_cost_per_hour": 0.0,
                    "total_cost_per_step": 0.0,
                    "vm_cpu_allocated": 0.0,
                    "vm_mem_allocated": 0.0,
                    "sla_violation": 0,
                    "cpu_utilization_pct": 0.0,
                    "mem_utilization_pct": 0.0,
                }
            )
            prev_alloc = {}
            continue

        plan = solve_vm_allocation(
            cpu_req,
            mem_req,
            vm_catalog,
            objective=objective,
            cpu_weight=1.0,
            mem_weight=0.5,
        )

        allocation = plan["allocation"]
        vm_cpu = plan["total_cpu"]
        vm_mem = plan["total_memory"]
        vm_cost = plan["total_cost"]
        vm_count = int(sum(allocation.values()))

        switch_cost = compute_switching_cost(prev_alloc, allocation, vm_catalog_map)
        total_cost = vm_cost + switch_cost
        vm_cost_step = vm_cost * STEP_HOURS
        total_cost_step = total_cost * STEP_HOURS + switch_cost

        cpu_util = min(100.0, (cpu_req / vm_cpu * 100.0)) if vm_cpu > 0 else (100.0 if cpu_req > 0 else 0.0)
        mem_util = min(100.0, (mem_req / vm_mem * 100.0)) if vm_mem > 0 else (100.0 if mem_req > 0 else 0.0)
        sla_violation = int(vm_cpu < cpu_req or vm_mem < mem_req)

        records.append(
            {
                "timestamp": row["timestamp"],
                "cpu_overflow_cores": cpu_req,
                "memory_overflow_gb": mem_req,
                "allocation": format_allocation(allocation),
                "vm_total_count": vm_count,
                "vm_cost_per_hour": vm_cost,
                "vm_cost_per_step": vm_cost_step,
                "switching_cost": switch_cost,
                "total_cost_per_hour": total_cost,
                "total_cost_per_step": total_cost_step,
                "vm_cpu_allocated": vm_cpu,
                "vm_mem_allocated": vm_mem,
                "sla_violation": sla_violation,
                "cpu_utilization_pct": cpu_util,
                "mem_utilization_pct": mem_util,
            }
        )
        prev_alloc = allocation

    return pd.DataFrame(records)


def build_reactive_schedule(
    requirements_df: pd.DataFrame,
    vm_catalog: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Solve LP per timestamp (reactive, no look-ahead). This mimics streaming decisions
    where only the current measurement is known.
    
    NOTE: This function uses cost-first plan. For dual-scenario output, use
    build_reactive_schedule_for_scenario() with scenario="cost" or "overload".
    """
    records = []
    prev_alloc: Dict[str, int] = {}
    vm_catalog_map = {vm["name"]: vm for vm in vm_catalog}

    for _, row in requirements_df.iterrows():
        cpu_req = float(row["cpu_overflow_cores"])
        mem_req = float(row["memory_overflow_gb"])

        if cpu_req <= 0 and mem_req <= 0:
            records.append(
                {
                    "timestamp": row["timestamp"],
                    "cpu_overflow_cores": cpu_req,
                    "memory_overflow_gb": mem_req,
                    "min_overload_plan": "Host only",
                    "min_overload_cost_per_hour": 0.0,
                    "min_cost_plan": "Host only",
                    "min_cost_cost_per_hour": 0.0,
                    "vm_plan": "Host only",
                    "vm_total_count": 0,
                    "vm_cost_per_hour": 0.0,
                    "vm_cost_per_step": 0.0,
                    "switching_cost": 0.0,
                    "total_cost_per_hour": 0.0,
                    "total_cost_per_step": 0.0,
                    "vm_cpu_allocated": 0.0,
                    "vm_mem_allocated": 0.0,
                    "sla_violation": 0,
                    "cpu_utilization_pct": 0.0,
                    "mem_utilization_pct": 0.0,
                }
            )
            prev_alloc = {}
            continue

        overload_plan = solve_vm_allocation(
            cpu_req,
            mem_req,
            vm_catalog,
            objective="capacity",
            cpu_weight=1.0,
            mem_weight=0.5,
        )
        cost_plan = solve_vm_allocation(
            cpu_req,
            mem_req,
            vm_catalog,
            objective="cost",
        )

        # Choose cost-first plan as enacted allocation
        allocation = cost_plan["allocation"]
        vm_cpu = cost_plan["total_cpu"]
        vm_mem = cost_plan["total_memory"]
        vm_cost = cost_plan["total_cost"]
        vm_count = int(sum(allocation.values()))

        switch_cost = compute_switching_cost(prev_alloc, allocation, vm_catalog_map)
        total_cost = vm_cost + switch_cost
        vm_cost_step = vm_cost * STEP_HOURS  # convert $/h to $/step
        total_cost_step = total_cost * STEP_HOURS + switch_cost  # switching is one-time at step

        cpu_util = min(100.0, (cpu_req / vm_cpu * 100.0)) if vm_cpu > 0 else (100.0 if cpu_req > 0 else 0.0)
        mem_util = min(100.0, (mem_req / vm_mem * 100.0)) if vm_mem > 0 else (100.0 if mem_req > 0 else 0.0)
        sla_violation = int(vm_cpu < cpu_req or vm_mem < mem_req)

        records.append(
            {
                "timestamp": row["timestamp"],
                "cpu_overflow_cores": cpu_req,
                "memory_overflow_gb": mem_req,
                "min_overload_plan": format_allocation(overload_plan["allocation"]),
                "min_overload_cost_per_hour": overload_plan["total_cost"],
                "min_cost_plan": format_allocation(cost_plan["allocation"]),
                "min_cost_cost_per_hour": cost_plan["total_cost"],
                "vm_plan": format_allocation(allocation),
                "vm_total_count": vm_count,
                "vm_cost_per_hour": vm_cost,
                "switching_cost": switch_cost,
                "total_cost_per_hour": total_cost,
                "vm_cost_per_step": vm_cost_step,
                "total_cost_per_step": total_cost_step,
                "vm_cpu_allocated": vm_cpu,
                "vm_mem_allocated": vm_mem,
                "sla_violation": sla_violation,
                "cpu_utilization_pct": cpu_util,
                "mem_utilization_pct": mem_util,
            }
        )
        prev_alloc = allocation

    return pd.DataFrame(records)


def compute_lp_metrics(schedule_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate LP results over the full test horizon.
    """
    if schedule_df.empty:
        return {
            "total_vm_cost": 0.0,  # $/h aggregated as hours in series freq
            "total_switching_cost": 0.0,
            "total_cost": 0.0,
            "total_cost_step": 0.0,
            "total_vms_sum": 0,
            "avg_vms": 0.0,
            "mean_cpu_utilization_pct": 0.0,
            "mean_mem_utilization_pct": 0.0,
            "sla_violations": 0,
            "sla_violation_rate": 0.0,
        }

    total_vm_cost = float(schedule_df["vm_cost_per_hour"].sum()) if "vm_cost_per_hour" in schedule_df else 0.0
    total_switching_cost = float(schedule_df["switching_cost"].sum()) if "switching_cost" in schedule_df else 0.0
    total_cost = float(schedule_df["total_cost_per_hour"].sum()) if "total_cost_per_hour" in schedule_df else total_vm_cost + total_switching_cost
    total_cost_step = float(schedule_df["total_cost_per_step"].sum()) if "total_cost_per_step" in schedule_df else 0.0

    total_vms_sum = int(schedule_df["vm_total_count"].sum()) if "vm_total_count" in schedule_df else 0
    avg_vms = float(schedule_df["vm_total_count"].mean()) if "vm_total_count" in schedule_df else 0.0

    mean_cpu_util = float(schedule_df["cpu_utilization_pct"].mean()) if "cpu_utilization_pct" in schedule_df else 0.0
    mean_mem_util = float(schedule_df["mem_utilization_pct"].mean()) if "mem_utilization_pct" in schedule_df else 0.0

    sla_violations = int(schedule_df["sla_violation"].sum()) if "sla_violation" in schedule_df else 0
    sla_violation_rate = float(schedule_df["sla_violation"].mean()) if "sla_violation" in schedule_df else 0.0

    return {
        "total_vm_cost": total_vm_cost,
        "total_switching_cost": total_switching_cost,
        "total_cost": total_cost,
        "total_cost_step": total_cost_step,
        "total_vms_sum": total_vms_sum,
        "avg_vms": avg_vms,
        "mean_cpu_utilization_pct": mean_cpu_util,
        "mean_mem_utilization_pct": mean_mem_util,
        "sla_violations": sla_violations,
        "sla_violation_rate": sla_violation_rate,
    }


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def run_planner(model_name: str = "unused", mode: str = "reactive"):
    """
    Reactive-only planner: uses ground-truth y_test as a streaming feed.
    Forecast-based planning is intentionally removed for LP in this mode.
    
    Outputs schedules for BOTH scenarios:
    - Minimize Overload (capacity-first)
    - Minimize Cost (cost-first)
    """
    print("=== VM Resource Planner (Reactive, no forecast) ===")
    print(f"Device: {DEVICE}")
    print("Mode: reactive (uses y_test streaming; model arg ignored)")

    vm_catalog = load_vm_catalog(VM_TYPES_FILE)
    data_df = load_ground_truth_df()
    requirements_df = convert_forecasts_to_requirements(data_df, HOST_SPEC)
    peak_summary = summarize_peak_plans(requirements_df, vm_catalog)
    
    # Build schedules for BOTH scenarios
    print("\nBuilding schedules for both scenarios...")
    schedule_overload = build_reactive_schedule_for_scenario(requirements_df, vm_catalog, scenario="overload")
    schedule_cost = build_reactive_schedule_for_scenario(requirements_df, vm_catalog, scenario="cost")
    
    metrics_overload = compute_lp_metrics(schedule_overload)
    metrics_cost = compute_lp_metrics(schedule_cost)

    # Persist outputs for OVERLOAD scenario
    schedule_overload_csv = RESULTS_DIR / "lp_schedule_overload.csv"
    schedule_overload.to_csv(schedule_overload_csv, index=False)
    
    # Persist outputs for COST scenario
    schedule_cost_csv = RESULTS_DIR / "lp_schedule_cost.csv"
    schedule_cost.to_csv(schedule_cost_csv, index=False)
    
    # Combined JSON report
    plan_payload = {
        "model": "vm_resource_planner_reactive",
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host_spec": HOST_SPEC,
        "peak_requirements": {
            "peak_cpu_overflow_cores": peak_summary["peak_cpu_overflow"],
            "peak_memory_overflow_gb": peak_summary["peak_memory_overflow"],
        },
        "scenarios": {
            "overload": {
                "peak_plan": peak_summary["minimize_overload_plan"],
                "metrics": metrics_overload,
            },
            "cost": {
                "peak_plan": peak_summary["minimize_cost_plan"],
                "metrics": metrics_cost,
            },
        },
    }

    results_path = RESULTS_DIR / "vm_resource_planning_reactive.json"
    save_results(plan_payload, str(results_path))

    # Print summary for BOTH scenarios
    print("\n" + "="*70)
    print("           SUMMARY: LP Reactive (Both Scenarios)")
    print("="*70)
    print(f"Peak CPU overflow (cores): {peak_summary['peak_cpu_overflow']:.2f}")
    print(f"Peak Memory overflow (GB): {peak_summary['peak_memory_overflow']:.2f}")
    
    print("\n--- Scenario: OVERLOAD (Minimize Resource Overload) ---")
    print(f"  Total VM cost ($/h sum):    {metrics_overload['total_vm_cost']:.4f}")
    print(f"  Total VMs (sum):            {metrics_overload['total_vms_sum']}")
    print(f"  Avg CPU util %:             {metrics_overload['mean_cpu_utilization_pct']:.2f}")
    print(f"  Avg Mem util %:             {metrics_overload['mean_mem_utilization_pct']:.2f}")
    print(f"  SLA violations:             {metrics_overload['sla_violations']} ({metrics_overload['sla_violation_rate']*100:.2f}%)")
    
    print("\n--- Scenario: COST (Optimize Operational Cost) ---")
    print(f"  Total VM cost ($/h sum):    {metrics_cost['total_vm_cost']:.4f}")
    print(f"  Total VMs (sum):            {metrics_cost['total_vms_sum']}")
    print(f"  Avg CPU util %:             {metrics_cost['mean_cpu_utilization_pct']:.2f}")
    print(f"  Avg Mem util %:             {metrics_cost['mean_mem_utilization_pct']:.2f}")
    print(f"  SLA violations:             {metrics_cost['sla_violations']} ({metrics_cost['sla_violation_rate']*100:.2f}%)")
    
    print("\n✓ Outputs:")
    print(f"  JSON report:          {results_path}")
    print(f"  LP Overload schedule: {schedule_overload_csv}")
    print(f"  LP Cost schedule:     {schedule_cost_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VM Resource Planner")
    parser.add_argument(
        "--mode",
        choices=["reactive"],
        default="reactive",
        help="Reactive only: use ground-truth y_test as streaming feed (no forecast).",
    )
    args = parser.parse_args()
    run_planner(mode=args.mode)



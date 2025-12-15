"""
Utility Functions for DRL VM Allocation
========================================

Contains helper functions for:
- Loading data and VM catalog
- Generating forecasts for RL state
- Parsing allocation strings
- Evaluating episodes
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from .config import DATA_CONFIG, HOST_SPEC, VM_TYPES_FILE
from model_utils import get_latest_model, load_model


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_vm_catalog(vm_file: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load VM catalog from JSON file.
    
    Args:
        vm_file: Path to VM types JSON file. If None, uses default.
        
    Returns:
        Dictionary of VM specifications
    """
    if vm_file is None:
        vm_file = get_project_root() / VM_TYPES_FILE
    else:
        vm_file = Path(vm_file)
    
    with open(vm_file, "r") as f:
        vm_catalog = json.load(f)
    
    return vm_catalog


def load_data(
    data_dir: Optional[str] = None,
    train_ratio: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split the cleaned data into train and test sets.
    
    Args:
        data_dir: Path to processed_data directory. If None, uses default.
        train_ratio: Fraction of data for training. If None, uses config default.
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if data_dir is None:
        data_dir = get_project_root() / "processed_data"
    else:
        data_dir = Path(data_dir)
    
    if train_ratio is None:
        train_ratio = DATA_CONFIG["train_ratio"]
    
    # Load cleaned data
    data_path = data_dir / "cleaned_data.csv"
    df = pd.read_csv(data_path)
    
    # Convert timestamp column
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Calculate split index
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    
    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df = df.iloc[n_train:].reset_index(drop=True)
    
    return train_df, test_df


def load_normalization_stats(
    data_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Load normalization statistics for inverse transformation.
    
    Args:
        data_dir: Path to processed_data directory.
        
    Returns:
        Dictionary with mean/std for each target
    """
    if data_dir is None:
        data_dir = get_project_root() / "processed_data"
    else:
        data_dir = Path(data_dir)
    
    stats_file = data_dir / "normalization_stats.json"
    
    if not stats_file.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {stats_file}\n"
            "Please run ETL pipeline to generate this file."
        )
    
    with open(stats_file, "r") as f:
        stats = json.load(f)
    
    return stats.get("targets", stats)


# ---------------------------------------------------------------------------
# Forecast utilities (per-target model loading)
# ---------------------------------------------------------------------------

# Default model mapping per target
TARGET_MODEL_MAP: Dict[str, str] = {
    "cpu_total_usage": "random_forest",
    "memory_usage_pct": "svr",
    "system_load": "random_forest",
}


def _load_split_xy(target: str, split: str = "test") -> Tuple[pd.Series, pd.DataFrame]:
    """
    Load X_split and y_split for a target from processed_data/<target>/.
    """
    base = get_project_root() / "processed_data" / target
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    y_path = base / f"y_{split}.csv"
    x_path = base / f"X_{split}.csv"
    if not y_path.exists() or not x_path.exists():
        raise FileNotFoundError(f"Missing processed data for target={target}, split={split}")
    y_split = pd.read_csv(y_path).squeeze()
    X_split = pd.read_csv(x_path)
    return y_split, X_split


def _predict_one_step_series(model, X_split: pd.DataFrame) -> np.ndarray:
    """
    One-step-ahead prediction for each row in X_split.
    """
    preds = model.predict(X_split)
    return np.asarray(preds).flatten()


def build_step_forecasts(
    target_model_map: Dict[str, str],
    horizon: int,
    split: str = "test",
) -> Dict[str, np.ndarray]:
    """
    Build per-step forecasts for each target using specified models.
    Each step's forecast is a vector of length `horizon` (broadcasted from 1-step prediction).

    Returns:
        Dict[target] -> np.ndarray shape (n_steps, horizon)
    """
    forecast_cache: Dict[str, np.ndarray] = {}

    for target, model_name in target_model_map.items():
        # Load data
        y_split, X_split = _load_split_xy(target, split)
        n_steps = len(X_split)

        # Load latest model for this target/model_name
        model_path = get_latest_model(models_dir="models", model_name=model_name, target=target)
        model, _meta = load_model(model_path)

        # One-step predictions for each row
        step_preds = _predict_one_step_series(model, X_split)

        # Broadcast each one-step pred to horizon
        fc_mat = np.repeat(step_preds[:, None], horizon, axis=1)
        forecast_cache[target] = fc_mat

    return forecast_cache


def compute_resource_requirements(
    row: pd.Series,
    host_spec: Optional[Dict] = None,
) -> Tuple[float, float, float, float]:
    """
    Compute CPU and memory requirements from a data row.
    
    Args:
        row: Pandas Series containing memory_usage_pct, cpu_total_usage, system_load
        host_spec: Host machine specifications
        
    Returns:
        Tuple of (cpu_required, mem_required, cpu_overflow, mem_overflow)
    """
    if host_spec is None:
        host_spec = HOST_SPEC
    
    # Extract values
    mem_pct = max(0, row.get("memory_usage_pct", 0))
    cpu_usage = max(0, row.get("cpu_total_usage", 0))
    system_load = max(0, row.get("system_load", 0))
    
    # Memory calculations
    mem_required_gb = host_spec["total_memory_gb"] * mem_pct / 100.0
    mem_threshold_gb = host_spec["total_memory_gb"] * host_spec["memory_threshold_pct"] / 100.0
    mem_overflow_gb = max(0, mem_required_gb - mem_threshold_gb)
    
    # CPU calculations
    cpu_required_cores = max(cpu_usage, system_load)
    cpu_threshold_cores = host_spec["total_cpu_cores"] * host_spec["cpu_threshold_pct"] / 100.0
    cpu_overflow_cores = max(0, cpu_required_cores - cpu_threshold_cores)
    
    return cpu_required_cores, mem_required_gb, cpu_overflow_cores, mem_overflow_gb


def generate_forecasts_for_rl(
    data: pd.DataFrame,
    current_idx: int,
    horizon: int = 12,
) -> np.ndarray:
    """
    Generate forecast-like features for RL state.
    
    In production, this would use the trained forecast models.
    For RL training, we use actual future values as "perfect forecasts".
    
    Args:
        data: Full dataset
        current_idx: Current timestep index
        horizon: Number of future steps to forecast
        
    Returns:
        Array of shape (horizon, 3) with [cpu, mem, load] forecasts
    """
    n_total = len(data)
    forecasts = []
    
    targets = ["cpu_total_usage", "memory_usage_pct", "system_load"]
    
    for h in range(horizon):
        future_idx = current_idx + h + 1
        
        if future_idx < n_total:
            row = data.iloc[future_idx]
            forecast = [row.get(t, 0) for t in targets]
        else:
            # If beyond data, repeat last known values
            last_row = data.iloc[-1]
            forecast = [last_row.get(t, 0) for t in targets]
        
        forecasts.append(forecast)
    
    return np.array(forecasts, dtype=np.float32)


def get_time_features(row: pd.Series) -> Tuple[int, int, bool]:
    """
    Extract time features from a data row.
    
    Args:
        row: Pandas Series with hour column or datetime
        
    Returns:
        Tuple of (hour, day_of_week, is_peak)
    """
    hour = int(row.get("hour", 0))
    
    # Try to get day of week from datetime
    if "datetime" in row and pd.notna(row["datetime"]):
        try:
            dt = pd.to_datetime(row["datetime"])
            day_of_week = dt.dayofweek
        except:
            day_of_week = 0
    else:
        day_of_week = 0
    
    # Define peak hours (e.g., 9 AM - 6 PM on weekdays)
    is_peak = (9 <= hour <= 18) and (day_of_week < 5)
    
    return hour, day_of_week, is_peak


def compute_vm_resources(
    vm_allocation: Dict[str, int],
    vm_catalog: Dict[str, Dict],
) -> Tuple[float, float, float]:
    """
    Compute total resources from VM allocation.
    
    Args:
        vm_allocation: VM allocation {vm_type: count}
        vm_catalog: VM specifications
        
    Returns:
        Tuple of (total_cpu, total_memory, total_cost)
    """
    total_cpu = 0.0
    total_memory = 0.0
    total_cost = 0.0
    
    for vm_type, count in vm_allocation.items():
        if vm_type in vm_catalog and count > 0:
            vm_spec = vm_catalog[vm_type]
            total_cpu += count * vm_spec.get("vcpus", 0)
            total_memory += count * vm_spec.get("memory_gb", 0)
            total_cost += count * vm_spec.get("cost_per_hour", 0)
    
    return total_cpu, total_memory, total_cost


def parse_allocation_string(allocation_str: str) -> Dict[str, int]:
    """
    Parse allocation string like "B2s×1, D2s_v3×2" to dictionary.
    
    Args:
        allocation_str: Allocation string
        
    Returns:
        Dictionary {vm_type: count}
    """
    allocation = {}
    
    if not allocation_str or allocation_str in ["No VMs", "Host only"]:
        return allocation
    
    # Split by comma and parse each part
    parts = allocation_str.split(",")
    
    for part in parts:
        part = part.strip()
        # Match pattern like "B2s×1" or "B2s x 1"
        match = re.match(r"(\w+)\s*[×x]\s*(\d+)", part)
        if match:
            vm_type = match.group(1)
            count = int(match.group(2))
            allocation[vm_type] = count
    
    return allocation


def format_allocation(vm_allocation: Dict[str, int]) -> str:
    """
    Format VM allocation dictionary to string.
    
    Args:
        vm_allocation: VM allocation {vm_type: count}
        
    Returns:
        Formatted string like "B2s:1, D2s_v3:2"
    """
    if not vm_allocation or all(v == 0 for v in vm_allocation.values()):
        return "No VMs"
    
    # Use ':' instead of '×' to avoid encoding issues in CSV
    parts = [f"{vm}:{count}" for vm, count in vm_allocation.items() if count > 0]
    return ", ".join(parts)


def evaluate_episode(
    allocations: List[Dict[str, int]],
    demands: List[Dict[str, float]],
    vm_catalog: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Evaluate a complete episode's performance.
    
    Args:
        allocations: List of VM allocations per timestep
        demands: List of resource demands per timestep
        vm_catalog: VM specifications
        
    Returns:
        Dictionary with evaluation metrics
    """
    n_steps = len(allocations)
    
    total_vm_cost = 0.0
    total_switching_cost = 0.0
    sla_violations = 0
    cpu_utils = []
    mem_utils = []
    total_vms = []
    
    prev_allocation = {}
    
    for t, (allocation, demand) in enumerate(zip(allocations, demands)):
        # Compute VM resources
        cpu_alloc, mem_alloc, vm_cost = compute_vm_resources(allocation, vm_catalog)
        total_vm_cost += vm_cost
        
        # Compute switching cost
        from .reward import compute_switching_cost
        switch_cost = compute_switching_cost(prev_allocation, allocation, vm_catalog)
        total_switching_cost += switch_cost
        
        # Check SLA violation
        cpu_req = demand.get("cpu_required", 0)
        mem_req = demand.get("mem_required", 0)
        
        if cpu_alloc < cpu_req or mem_alloc < mem_req:
            sla_violations += 1
        
        # Compute utilization
        if cpu_alloc > 0:
            cpu_utils.append(min(100, cpu_req / cpu_alloc * 100))
        if mem_alloc > 0:
            mem_utils.append(min(100, mem_req / mem_alloc * 100))
        
        # Count VMs
        total_vms.append(sum(allocation.values()))
        
        prev_allocation = allocation.copy()
    
    return {
        "n_steps": n_steps,
        "total_vm_cost": total_vm_cost,
        "total_switching_cost": total_switching_cost,
        "total_cost": total_vm_cost + total_switching_cost,
        "sla_violations": sla_violations,
        "sla_violation_rate": sla_violations / n_steps if n_steps > 0 else 0,
        "mean_cpu_utilization": np.mean(cpu_utils) if cpu_utils else 0,
        "mean_mem_utilization": np.mean(mem_utils) if mem_utils else 0,
        "mean_vms_used": np.mean(total_vms) if total_vms else 0,
        "total_vms_sum": sum(total_vms),
    }


def action_to_allocation(
    action: np.ndarray,
    vm_types: List[str],
) -> Dict[str, int]:
    """
    Convert action array to VM allocation dictionary.
    
    Args:
        action: Array of VM counts per type
        vm_types: List of VM type names
        
    Returns:
        Dictionary {vm_type: count}
    """
    allocation = {}
    for i, vm_type in enumerate(vm_types):
        count = int(max(0, action[i]))
        if count > 0:
            allocation[vm_type] = count
    return allocation


def allocation_to_action(
    allocation: Dict[str, int],
    vm_types: List[str],
) -> np.ndarray:
    """
    Convert VM allocation dictionary to action array.
    
    Args:
        allocation: Dictionary {vm_type: count}
        vm_types: List of VM type names
        
    Returns:
        Array of VM counts per type
    """
    action = np.zeros(len(vm_types), dtype=np.int32)
    for i, vm_type in enumerate(vm_types):
        action[i] = allocation.get(vm_type, 0)
    return action


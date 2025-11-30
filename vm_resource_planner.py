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
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import linprog

from model_utils import get_latest_model, load_model, save_results

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
VM_TYPES_FILE = PROJECT_ROOT / "VMs_type.json"
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Host machine specification (can be adjusted)
HOST_SPEC = {
    "total_cpu_cores": 16,
    "total_memory_gb": 64,
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


def generate_forecasts(model_paths: Dict[str, Path], model_name: str) -> pd.DataFrame:
    """Run inference and build a forecast DataFrame for the selected model."""
    model_type = SUPPORTED_MODELS.get(model_name)
    if model_type is None:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_type == "hybrid":
        return _generate_forecasts_hybrid(model_paths)
    elif model_type == "stats":
        return _generate_forecasts_stats(model_paths, model_name)
    else:
        return _generate_forecasts_ml(model_paths, model_name)


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

    mem_pct = df["memory_usage_pct"].clip(lower=0)
    cpu_pct = df["cpu_total_usage"].clip(lower=0)
    system_load = df["system_load"].clip(lower=0)

    mem_total_gb = host_spec["total_memory_gb"] * mem_pct / 100.0
    mem_threshold_gb = (
        host_spec["total_memory_gb"] * host_spec["memory_threshold_pct"] / 100.0
    )
    mem_overflow_gb = np.maximum(0.0, mem_total_gb - mem_threshold_gb)

    cpu_pct_cores = host_spec["total_cpu_cores"] * cpu_pct / 100.0
    cpu_threshold_cores = (
        host_spec["total_cpu_cores"] * host_spec["cpu_threshold_pct"] / 100.0
    )
    cpu_required_cores = np.maximum(cpu_pct_cores, system_load)
    cpu_overflow_cores = np.maximum(0.0, cpu_required_cores - cpu_threshold_cores)

    df["memory_required_gb"] = mem_total_gb
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
        c = [vm["cost_per_hour"] for vm in vm_catalog]
    else:
        c = [
            cpu_weight * vm["vcpus"] + mem_weight * vm["memory_gb"]
            for vm in vm_catalog
        ]

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


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def run_planner(model_name: str):
    print("=== VM Resource Planner ===")
    print(f"Device: {DEVICE}")
    print(f"Model: {model_name}")

    vm_catalog = load_vm_catalog(VM_TYPES_FILE)
    model_paths = load_latest_models(model_name)
    forecast_df = generate_forecasts(model_paths, model_name)

    requirements_df = convert_forecasts_to_requirements(forecast_df, HOST_SPEC)
    peak_summary = summarize_peak_plans(requirements_df, vm_catalog)
    schedule_df = build_schedule(requirements_df, vm_catalog)

    # Persist outputs
    plan_payload = {
        "model": "vm_resource_planner",
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host_spec": HOST_SPEC,
        "peak_requirements": {
            "peak_cpu_overflow_cores": peak_summary["peak_cpu_overflow"],
            "peak_memory_overflow_gb": peak_summary["peak_memory_overflow"],
        },
        "scenarios": {
            "minimize_overload": peak_summary["minimize_overload_plan"],
            "minimize_cost": peak_summary["minimize_cost_plan"],
        },
        "schedule": schedule_df.to_dict(orient="records"),
    }

    results_path = RESULTS_DIR / "vm_resource_planning.json"
    schedule_records = schedule_df.copy()
    schedule_records["timestamp"] = schedule_records["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    plan_payload["schedule"] = schedule_records.to_dict(orient="records")
    save_results(plan_payload, str(results_path))

    schedule_csv = RESULTS_DIR / "vm_schedule.csv"
    schedule_df.to_csv(schedule_csv, index=False)

    print("\n=== Summary ===")
    print(f"Peak CPU overflow (cores): {peak_summary['peak_cpu_overflow']:.2f}")
    print(f"Peak Memory overflow (GB): {peak_summary['peak_memory_overflow']:.2f}")
    print("\nScenario: Minimize Resource Overload")
    print(f"  Allocation: {format_allocation(peak_summary['minimize_overload_plan']['allocation'])}")
    print(f"  Total cost $/h: {peak_summary['minimize_overload_plan']['total_cost']:.4f}")
    print("\nScenario: Optimize Operational Cost")
    print(f"  Allocation: {format_allocation(peak_summary['minimize_cost_plan']['allocation'])}")
    print(f"  Total cost $/h: {peak_summary['minimize_cost_plan']['total_cost']:.4f}")
    print(f"\n✓ JSON report: {results_path}")
    print(f"✓ Schedule CSV: {schedule_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VM Resource Planner")
    parser.add_argument(
        "--model",
        choices=list(SUPPORTED_MODELS.keys()),
        default=DEFAULT_MODEL_NAME,
        help="Model name to use for forecasting.",
    )
    args = parser.parse_args()
    run_planner(args.model)



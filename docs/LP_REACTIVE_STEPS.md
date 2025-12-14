## LP Reactive Mode (No Forecast) — How It Works

We run Linear Programming allocation using *measured* values only (no model forecast). `y_test` is treated as a streaming feed: at each timestamp we only use the measurement at that time, not future values.

### What’s in code now
- Only **reactive** mode for LP (`--mode reactive`).
- `load_ground_truth_df()` builds a DataFrame from `y_test` with timestamps.
- `build_reactive_schedule()` solves LP **per timestamp** (no bucket/look-ahead), applies VM switching cost, and records utilization/SLA.
- Outputs:
  - JSON: `forecast_result/vm_resource_planning_reactive.json`
  - CSV: `forecast_result/vm_schedule_reactive.csv`
  - Metrics included: total VM cost, switching cost, total cost, total VMs, avg utilization, SLA violations.

### Flow (reactive)
1. Load VM catalog (includes `cost_per_hour` and `switching_cost`).
2. Stream `y_test` for the three targets as if they arrive in real time.
3. Convert measurements → requirements (`cpu_overflow_cores`, `memory_overflow_gb`) vs host thresholds.
4. For **each timestamp**, solve LP (cost-first plan enacted), compute switching cost from last allocation, utilization, SLA flag.
5. Save per-timestep schedule (no resampling/bucketing) and aggregate metrics.

### How to run
```bash
python vm_resource_planner.py --mode reactive
```

### Rationale
- In production, future measurements are unknown; reactive mode matches that by using only current observations.
- Forecast-based LP has been removed for this flow to avoid “oracle” look-ahead.***


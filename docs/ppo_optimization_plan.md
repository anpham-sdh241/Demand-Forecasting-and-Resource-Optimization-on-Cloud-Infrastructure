# PPO vs LP: Findings & Optimization Plan

## Current Findings (from `ppo_schedule_test_cost.csv` / `ppo_schedule_test_overload.csv`)
- PPO is **always provisioning VMs** even when `cpu_overflow_cores = 0` and `mem_overflow_gb = 0`.
- Costs are orders of magnitude higher than LP (`vm_schedule_notebook.csv` shows host-only or a single `B2s` for sparse overflow).
- SLA violations are zero (good), but cost is the dominant issue.
- Evaluation is per 30s step; LP is bucketed 30 minutes, so PPO looks even more expensive when summed.

## Diagnosis
- Reward weights for cost/switching are too low relative to VM cost; over-provisioning is not penalized strongly.
- No explicit penalty for “overflow=0 but VMs>0”.
- Efficiency bonus does not discourage over-provision when no overflow.
- Action space allows large allocations regardless of overflow size.
- Evaluation is per-step (30s) and not bucketed to 30 minutes like LP.

## Concrete Optimization Plan

### 1) Reward & Penalties
- Cost-first scenario: increase cost/switching weights  
  - `RewardConfig.cost_scenario`: set `delta = 2.0`, `gamma = 0.8`.
- Add over-provision penalty (when overflow=0 and VMs>0):  
  - Penalty = `overprov_penalty_cpu * total_vm_vcpus + overprov_penalty_mem * total_vm_mem_gb`  
  - Suggested: `overprov_penalty_cpu = 0.05`, `overprov_penalty_mem = 0.01`.
- Adjust efficiency bonus:
  - If overflow=0 and VMs=0 → bonus +0.5 (already fine).
  - If overflow=0 and VMs>0 → penalty (e.g., -0.5) to discourage idle VMs.

### 2) Action Constraints / Post-processing
- Hard rule in evaluation: if `cpu_overflow_cores == 0` and `mem_overflow_gb == 0` → `allocation = Host only` (VM counts = 0, cost = 0, switching = 0). This mirrors LP behavior.
- Optional: cap VM counts to cover only the overflow with a small safety margin (e.g., 20%).

### 3) Training Settings
- Train longer: `total_timesteps = 1_000_000` for both scenarios.
- Keep `episode_length = 480` (4h) but ensure enough updates: with `n_steps=2048`, expect ~488 updates at 1M steps.
- Increase `ent_coef` slightly (e.g., 0.02) only if exploration seems insufficient; otherwise keep 0.01.
- Keep `horizon = 18` (9 minutes) as decided from evaluation.
- Use `n_envs = 8` if resources permit to stabilize gradients.

### 4) Evaluation Alignment with LP (30-minute buckets)
- After rollout, resample to 30-minute buckets to match LP:
  - For costs: sum within bucket.
  - For demand/overflow: max within bucket.
  - For SLA: sum violations; rate = mean.
- Save additional files:  
  - `forecast_result/ppo_schedule_30min_<scenario>.csv`  
  - `forecast_result/ppo_vs_lp_comparison_30min.json`

### 5) Code Change Checklist (high level)
- `rl/reward.py`: add over-provision penalty and tweak cost/switching weights in `cost_scenario`.
- `eval_ppo.py`: post-process per-step outputs to force host-only when overflow=0; add 30-minute resample outputs.
- `train_ppo.py` config: set `total_timesteps=1_000_000`, `n_envs=8` (if available), and updated reward weights.

### 6) Quick Rationale vs LP
- LP turns on VMs only when overflow > 0 in 30-minute buckets; PPO must emulate this by:
  - Strong cost/switching penalties,
  - Explicit penalty for idle VMs when overflow=0,
  - Optional hard rule at inference: host-only if no overflow.

### 7) Suggested Next Run
```bash
# Update reward weights & over-provision penalty (code changes required)
# Then retrain both scenarios:
python train_ppo.py --scenario both --timesteps 1000000 --n-envs 8

# Evaluate with post-processing and 30-min buckets:
python eval_ppo.py --scenario both
# (after code changes to add host-only rule + 30min resample)
```

Applying the above should drastically reduce VM usage when overflow=0 and bring PPO costs closer to, or below, the LP baseline while preserving zero SLA violations.


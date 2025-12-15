# PPO vs LP: Findings & Optimization Plan

## Current Findings (from `ppo_schedule_test_cost.csv` / `ppo_schedule_test_overload.csv`)
- PPO is **always provisioning VMs** even when `cpu_overflow_cores = 0` and `mem_overflow_gb = 0`.
- Costs are orders of magnitude higher than LP (`vm_schedule_notebook.csv` shows host-only or a single `B2s` for sparse overflow).
- SLA violations are zero (good), but cost is the dominant issue.
- Evaluation is per 30s step; LP is bucketed 30 minutes, so PPO looks even more expensive when summed.

## Diagnosis
- Reward weights for cost/switching are too low relative to VM cost; over-provisioning is not penalized strongly.
- No explicit penalty for "overflow=0 but VMs>0".
- Efficiency bonus does not discourage over-provision when no overflow.
- Action space allows large allocations regardless of overflow size.
- Evaluation is per-step (30s) and not bucketed to 30 minutes like LP.

## Concrete Optimization Plan (UPDATED 2025-12-16)

### 1) Reward & Penalties ✅ IMPLEMENTED
- Cost-first scenario: increase cost/switching weights  
  - `RewardConfig.cost_scenario`: `delta = 2.0`, `gamma = 0.8`.
- **Over-provision penalty** (when overflow=0 and VMs>0):  
  - `overprov_penalty_per_vcpu = 0.5` (10x stronger than before)
  - `overprov_penalty_per_gb = 0.2` (20x stronger than before)
- **Efficiency bonus improvements**:
  - If overflow=0 and VMs=0 → bonus **+1.0** (perfect)
  - If overflow=0 and VMs>0 → penalty proportional to waste (up to -1.0)
  - Smoother gradient for utilization ranges (50-70%, 70-80%, 80-95%)

### 2) Action Constraints / Post-processing ✅ ALREADY IN EVAL
- Hard rule in evaluation: if `cpu_overflow_cores == 0` and `mem_overflow_gb == 0` → `allocation = Host only`.
- This is already implemented in `eval_ppo.py` rollout function.

### 3) Training Settings ✅ UPDATED
- Train longer: `total_timesteps = 1_000_000` for both scenarios.
- Episode length: `episode_length = 480` (4h) for training stability.
- **Forecast horizon: `horizon = 120` steps (60 minutes look-ahead)** - UPDATED from 18 steps (9 min).
- Entropy coefficient: `ent_coef = 0.01` (default).
- Use `n_envs = 8` if resources permit.

### 4) Evaluation Output (Per-Step)
- PPO evaluation outputs results per-step (not bucketed by 30 minutes).
- Each step represents a 30-second interval matching the raw data granularity.
- Output files:
  - `forecast_result/ppo_schedule_test_<scenario>.csv` - Per-step schedule
  - `forecast_result/ppo_schedule_30min_<scenario>.csv` - 30-minute aggregated
- LP vs PPO comparison is done in `lp_vs_ppo_comparison.ipynb`.

### 5) Code Changes Summary

| File | Change |
|------|--------|
| `rl/config.py` | `horizon = 120` (60 min), `overprov_penalty = 0.5/0.2` |
| `rl/reward.py` | Improved efficiency bonus with smoother gradients |
| `eval_ppo.py` | Post-process forces host-only when overflow=0 |

### 6) Quick Rationale vs LP
- LP turns on VMs only when overflow > 0 in 30-minute buckets; PPO must emulate this by:
  - Strong cost/switching penalties,
  - **Strong penalty for idle VMs when overflow=0** (now 10-20x stronger),
  - Hard rule at inference: host-only if no overflow.

### 7) Train New Models

```bash
# Delete old models (they use horizon=18, incompatible with new horizon=120)
rm rl_models/ppo_*.zip rl_models/ppo_*.pkl

# Retrain both scenarios with new settings:
python train_ppo.py --scenario both --timesteps 1000000 --n-envs 8

# Evaluate:
python eval_ppo.py --scenario both
```

### 8) Expected Improvements
With the new settings, PPO should:
- Allocate VMs **only when overflow > 0**
- Use **minimal VMs** to cover overflow (not 10x B2s for 0.24 cores overflow!)
- Have costs **closer to LP baseline**
- Maintain **zero SLA violations**

The 60-minute forecast horizon gives PPO more context to plan VM allocation ahead.

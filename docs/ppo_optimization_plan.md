# PPO vs LP: Findings & Optimization Plan

## Current Findings (2025-12-16)

### Comparison Results (17,150 steps = Full Test Set)

| Metric | LP (OVERLOAD) | PPO (OVERLOAD) | LP (COST) | PPO (COST) |
|--------|---------------|----------------|-----------|------------|
| Total VMs | 2,738 | 6,383 | 4,015 | 2,420 |
| Max VMs/step | 1 | **36** ❌ | 7 | 16 |
| Steps with VMs | 2,738 | 394 | 2,738 | 354 |
| CPU Util % | 5.6% | **1.21%** ❌ | 53.6% | **7.02%** ❌ |
| Total Cost ($/h) | $4,205 | $2,784 | **$167** | $291 |
| SLA Violations | 0 | 0 | **0** | 40 ❌ |

### Key Issues
1. **Over-provisioning**: PPO allocates 16-36 VMs at once when it does allocate
2. **Under-utilization**: CPU utilization 1-7% (vs LP's 5-53%)
3. **SLA Violations**: 40 violations in COST scenario
4. **Sparse allocation**: PPO only allocates VMs in 2-3% of steps, but over-provisions when it does

---

## Diagnosis

| Problem | Root Cause |
|---------|------------|
| Over-provisioning | Weak VM cost penalty (delta too low) |
| Under-utilization | No efficiency bonus for right-sizing |
| SLA violations | alpha=0.3 too low in COST scenario |
| Sparse allocation | Agent learned to "burst" instead of "trickle" |

---

## Training Parameters Explained

### 1. `total_timesteps` - Training Duration

```
total_timesteps = 2,000,000 (minimum with episode_length=2880)
                = 5,000,000 (default - recommended) ✅
                = 10,000,000 (best quality, ~5-6 hours)
```

**Calculation**:
```
n_envs = 8 parallel environments
episode_length = 2880 steps/episode (training - daily pattern) | 17150 steps (evaluation)
n_steps = 2048 steps/update

Updates = total_timesteps / (n_envs × n_steps) = 5M / (8 × 2048) = 305 updates
Episodes (training) ≈ total_timesteps / 2880 = 5M / 2880 = 1,736 episodes
```

**Guideline** (with episode_length=2880):
| timesteps | Episodes | Quality | Training Time |
|-----------|----------|---------|---------------|
| 2M | 694 | Minimum | ~1 hour |
| 5M | 1,736 | Recommended ✅ | ~2-3 hours |
| 10M | 3,472 | Best | ~5-6 hours |

### 2. `learning_rate` - Learning Speed

```python
learning_rate = 3e-4  # Default, good for most cases
              = 1e-4  # More stable, slower learning
              = 1e-3  # Faster but may diverge
```

**When to adjust**:
- Training loss oscillating → decrease learning_rate
- Training too slow → increase learning_rate

### 3. `n_steps` - Steps Before Policy Update

```python
n_steps = 2048  # Default
        = 512   # More frequent updates (less stable)
        = 4096  # Less frequent (more stable, slower)
```

**Trade-off**: More steps = more stable gradients but slower learning

### 4. `gamma` - Discount Factor

```python
gamma = 0.99  # Default - values future rewards highly
      = 0.95  # More short-term focused
      = 0.999 # Very long-term focused
```

**For VM allocation**: 0.99 is good (balance short and long-term costs)

### 5. `ent_coef` - Exploration vs Exploitation

```python
ent_coef = 0.01  # Default
         = 0.1   # More exploration (random actions)
         = 0.001 # Less exploration (exploit learned policy)
```

### 6. `episode_length` - Episode Duration

```python
# TRAINING: Choose based on data pattern
episode_length = 480   # 4 hours - use if NO daily pattern
               = 1440  # 12 hours
               = 2880  # 24 hours - use if data has DAILY pattern ✅ RECOMMENDED

# EVALUATION: Use full test set
episode_length = 17150  # Full test set (~6 days = 17150 × 30s)
```

**Choosing episode_length for training:**

| Data Pattern | episode_length | Reason |
|--------------|----------------|--------|
| Daily cycle | **2880** (24h) | Agent sees full day pattern |
| Weekly cycle | 2880 + more timesteps | Learn daily, generalize to weekly |
| No pattern | 480 (4h) | Faster learning, more episodes |

**Current config**: `episode_length = 2880` (daily pattern detected)
- 5M timesteps / 2880 = 1,736 episodes
- Agent learns: morning peak, afternoon dip, night low, etc.

**Note**: In `drl_ppo_vm_allocation.ipynb`, `episode_length` is overridden to 17150 for evaluation only.

### 7. `horizon` - Forecast Look-ahead

```python
horizon = 120  # 60 minutes look-ahead (default)
        = 60   # 30 minutes
        = 240  # 2 hours (larger state space)
```

---

## Optimization Plan

### Phase 1: Reward Tuning ✅ DONE

**Changes to `rl/config.py`**:

```python
# OVERLOAD scenario
def overload_scenario():
    return RewardConfig(
        alpha=1.0,    # SLA penalty
        beta=0.7,     # Overflow penalty
        gamma=0.2,    # Switching cost
        delta=0.5,    # VM cost (increased from 0.1)
        epsilon=0.3,  # Efficiency bonus (increased from 0.1)
    )

# COST scenario
def cost_scenario():
    return RewardConfig(
        alpha=1.0,    # SLA penalty (increased from 0.3)
        beta=0.3,     # Overflow penalty
        gamma=0.8,    # Switching cost
        delta=2.0,    # VM cost
        epsilon=0.3,  # Efficiency bonus (increased from 0.1)
    )
```

### Phase 2: Training with Daily Pattern Config 🔄 TODO

```bash
# Delete old models (incompatible with new config)
rm rl_models/ppo_*.zip rl_models/ppo_*.pkl

# Train with 5M steps + episode_length=2880 (24h daily cycle)
# This allows agent to learn daily patterns (peak hours, etc.)
python train_ppo.py --scenario both --timesteps 5000000 --n-envs 8

# Evaluate on FULL test set (17150 steps = ~6 days)
python eval_ppo.py --scenario both --episode-length 17150
```

### Phase 3: Hyperparameter Tuning (if needed)

If still under-performing after Phase 2:

| Adjustment | When to use |
|------------|-------------|
| Increase `learning_rate` to 1e-3 | Training too slow |
| Decrease `learning_rate` to 1e-4 | Training unstable |
| Increase `overprov_penalty` to 1.0 | Still over-provisioning |
| Increase `total_timesteps` to 5M | Policy not converging |

---

## Expected Improvements After Retraining

| Metric | Current PPO | Target |
|--------|-------------|--------|
| Max VMs/step | 16-36 | ≤3 |
| CPU Util % | 1-7% | >30% |
| SLA Violations | 40 | 0 |
| Total Cost | $291 | <$200 |

---

## Files Changed

| File | Changes |
|------|---------|
| `rl/config.py` | Updated reward weights for both scenarios |
| `rl/utils.py` | Fixed `format_allocation` to support both `:` and `×` separators |
| `eval_ppo.py` | Host-only label for 0 VMs allocation |
| `lp_vs_ppo_comparison.ipynb` | Fixed VM count extraction for PPO format |
| `vm_resource_planner.py` | Fixed LP to output different results for OVERLOAD vs COST |

---

## How to Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir=tensorboard_logs

# Watch for:
# - ep_rew_mean: Should increase over time
# - policy_loss: Should decrease and stabilize
# - entropy_loss: Should decrease slowly (not too fast)
```

---

## Quick Commands

```bash
# Full retraining pipeline (with daily pattern config)
rm rl_models/ppo_*.zip rl_models/ppo_*.pkl
python train_ppo.py --scenario both --timesteps 5000000 --n-envs 8

# Evaluate with full test set (17150 steps)
python eval_ppo.py --scenario both --episode-length 17150

# Run comparison notebook
jupyter notebook lp_vs_ppo_comparison.ipynb
```

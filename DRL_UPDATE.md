# DRL Update Plan (PPO for VM Allocation)

## Goals
- Train PPO agent to allocate VMs given forecasted CPU/RAM/load.
- Support two evaluation scenarios:
  1) Minimize Resource Overload (CPU/RAM priority)
  2) Optimize Operational Cost (cost/switching priority)
- Compare DRL vs LP baseline (`vm_resource_planner.py`).

## Files to Add / Update
- Update `requirements.txt`: add `gymnasium`, `stable-baselines3`, `tensorboard`, `shimmy` (or `ray[rllib]` if dùng RLlib).
- Update `VMs_type.json`: thêm `switching_cost` (hoặc start/stop cost nếu cần chi tiết).
- New folder `rl/`:
  - `rl/__init__.py`
  - `rl/environment.py` (Gym Env `VMAllocationEnv`: state = forecast + current_vms + time features; action = VM counts; reward = -cost - switching - SLA/overflow + efficiency bonus)
  - `rl/reward.py` (hàm reward, chứa trọng số cho 2 kịch bản)
  - `rl/config.py` (hyperparams PPO)
  - `rl/utils.py` (load data, forecast wrapper, parse allocation, evaluate)
- New scripts:
  - `train_ppo.py` (train, log TensorBoard, save checkpoint `.zip`)
  - `eval_ppo.py` (load checkpoint, rollout test, so sánh với LP, xuất CSV)
- New folder `rl_models/` để lưu checkpoint (ví dụ `ppo_overload.zip`, `ppo_cost.zip`).

## Data Used for Training
- `processed_data/cleaned_data.csv` (actual demand per 30s).
- Forecast models trong `models/` để sinh forecast làm state input.
- Env tự quản lý `current_vms` (không cần cột hiện trạng VM trong data).

### Cách chia train / test
- Train: dùng đoạn đầu của `cleaned_data.csv` (ví dụ 70-80% thời gian, theo thứ tự thời gian).
- Test: dùng đoạn cuối (20-30%) để rollout và so sánh với LP.
- Mỗi episode: một cửa sổ thời gian liên tục (ví dụ 1 ngày = 288 bước x 5 phút, hoặc 1 giờ = 120 bước x 30 giây) tùy thiết lập.
- Có thể random hóa điểm bắt đầu episode trong tập train để tăng đa dạng.

### Dữ liệu đầu vào Env
- Actual demand: lấy trực tiếp từ `cleaned_data.csv` tại timestep.
- Forecast: sinh từ models trong `models/` cho horizon (ví dụ 12 bước tới) để đưa vào state.
- Time features: hour, day-of-week, is_peak.
- VM state: do Env tự giữ (`current_vms`), khởi tạo 0 (hoặc từ LP baseline nếu muốn).

## Reward Config (gợi ý)
```
reward = - α * sla_violation
         - β * overflow_cpu_ram
         - γ * switching_cost
         - δ * vm_cost
         + ε * efficiency_bonus
```
- Scenario 1 (Overload-first): α=1.0, β=0.7, γ=0.2, δ=0.1, ε=0.1
- Scenario 2 (Cost-first):     α=0.3, β=0.2, γ=0.5, δ=1.0, ε=0.1

## Testing & Outputs
- LP baseline output: `forecast_result/vm_schedule.csv`.
- PPO output (per scenario): `forecast_result/ppo_schedule_test_<scenario>.csv` with columns:
  - `timestamp`
  - `allocation` (e.g. `B2s×1, D2s_v3×0, ...`)
  - `vm_cost_per_hour`
  - `switching_cost`
  - `total_cost_per_hour`
  - `cpu_allocated_cores`, `mem_allocated_gb`
  - `cpu_required_cores`, `mem_required_gb`
  - `cpu_utilization_pct`, `mem_utilization_pct`
  - `sla_violation_flag`

### Compare DRL vs LP on
- Total VMs used (sum or average over horizon).
- Resource utilization (mean CPU/RAM utilization).
- Operational cost + switching cost; SLA violations.

## Evaluation Metrics (gợi ý)
- **Cost**: tổng `total_cost_per_hour` (PPO) vs `min_cost_cost_per_hour` (LP).
- **Switching cost**: tổng switching (PPO có, LP có thể ~0 nếu không tính).
- **SLA violations**: số bước thiếu CPU/RAM (overflow>0).
- **Utilization**: trung bình `cpu_utilization_pct`, `mem_utilization_pct`.
- **Total VMs used**: tổng hoặc trung bình số VM ở mỗi bước.


## Minimal Flow
1) Cập nhật `requirements.txt`, `VMs_type.json`.
2) Thêm `rl/` modules, `train_ppo.py`, `eval_ppo.py`, tạo `rl_models/`.
3) Train hai kịch bản riêng (overload-first, cost-first) → lưu hai checkpoint.
4) Rollout trên tập test → sinh 2 CSV kết quả PPO.
5) So sánh với LP CSV theo 3 tiêu chí nêu trên.


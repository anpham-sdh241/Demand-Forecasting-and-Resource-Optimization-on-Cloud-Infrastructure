"""
Configuration for PPO VM Allocation
====================================

Contains hyperparameters and reward configurations for two scenarios:
1. SCENARIO_OVERLOAD: Prioritizes minimizing SLA violations and resource overflow
2. SCENARIO_COST: Prioritizes minimizing operational and switching costs
"""

from dataclasses import dataclass, field
from typing import Dict

# Scenario identifiers
SCENARIO_OVERLOAD = "overload"
SCENARIO_COST = "cost"


@dataclass
class RewardConfig:
    """
    Reward weighting configuration.
    
    reward = - α * sla_violation
             - β * overflow_cpu_ram
             - γ * switching_cost
             - δ * vm_cost
             + ε * efficiency_bonus
    """
    alpha: float = 1.0    # SLA violation weight
    beta: float = 0.7     # CPU/RAM overflow weight
    gamma: float = 0.2    # Switching cost weight
    delta: float = 0.1    # VM cost weight
    epsilon: float = 0.1  # Efficiency bonus weight
    
    # Penalty scales
    sla_penalty_per_core: float = 10.0      # Penalty per core of SLA violation
    sla_penalty_per_gb: float = 5.0         # Penalty per GB of memory SLA violation
    overflow_penalty_per_core: float = 5.0  # Penalty per core overflow
    overflow_penalty_per_gb: float = 2.5    # Penalty per GB overflow
    # Over-provision penalty (when overflow=0 but VMs>0)
    overprov_penalty_per_vcpu: float = 0.05
    overprov_penalty_per_gb: float = 0.01
    
    @classmethod
    def overload_scenario(cls) -> "RewardConfig":
        """Scenario 1: Minimize Resource Overload (CPU/RAM priority)"""
        return cls(
            alpha=1.0,
            beta=0.7,
            gamma=0.2,
            delta=0.1,
            epsilon=0.1,
        )
    
    @classmethod
    def cost_scenario(cls) -> "RewardConfig":
        """Scenario 2: Optimize Operational Cost (cost/switching priority)"""
        return cls(
            alpha=0.3,
            beta=0.2,
            gamma=0.8,
            delta=2.0,
            epsilon=0.1,
        )


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters"""
    
    # Training parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048           # Steps per update
    batch_size: int = 64          # Minibatch size
    n_epochs: int = 10            # Epochs per update
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    clip_range: float = 0.2       # PPO clip range
    clip_range_vf: float = None   # VF clip range (None = no clipping)
    ent_coef: float = 0.01        # Entropy coefficient
    vf_coef: float = 0.5          # Value function coefficient
    max_grad_norm: float = 0.5    # Gradient clipping
    
    # Environment parameters
    # Data sampling: every 30 seconds
    # episode_length = số bước trong 1 episode
    #   - 2880 steps × 30s = 24 giờ (1 ngày)
    #   - 1440 steps × 30s = 12 giờ
    #   - 480 steps × 30s = 4 giờ
    episode_length: int = 480     # 480 steps = 4 giờ mô phỏng
    
    # horizon = số bước forecast phía trước
    #   - 9 phút = 9 × 60 / 30 = 18 steps
    #   - 12 phút = 12 × 60 / 30 = 24 steps
    #   - 6 phút = 6 × 60 / 30 = 12 steps
    horizon: int = 18             # 18 steps × 30s = 9 phút forecast
    
    # Training duration
    total_timesteps: int = 1_000_000
    
    # Logging
    tensorboard_log: str = "./tensorboard_logs/"
    log_interval: int = 10        # Log every N updates
    
    # Checkpointing
    save_freq: int = 10_000       # Save checkpoint every N steps
    
    def to_sb3_kwargs(self) -> Dict:
        """Convert to Stable-Baselines3 PPO kwargs"""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "tensorboard_log": self.tensorboard_log,
        }


# Host machine specification (same as vm_resource_planner.py)
HOST_SPEC = {
    "total_cpu_cores": 1,
    "total_memory_gb": 4,
    "cpu_threshold_pct": 70,     # Begin offloading after 70% CPU usage
    "memory_threshold_pct": 75,  # Begin offloading after 75% memory usage
}

# Data configuration
DATA_CONFIG = {
    "series_freq": "30S",
    "start_timestamp": "2024-01-01 00:00:00",
    "train_ratio": 0.8,          # 80% train, 20% test
}

# VM Types file path
VM_TYPES_FILE = "VMs_type.json"

# Max VMs per type (for action space bounds)
MAX_VMS_PER_TYPE = 10


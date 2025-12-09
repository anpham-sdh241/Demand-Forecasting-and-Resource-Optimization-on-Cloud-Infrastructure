"""
Reinforcement Learning Module for VM Allocation
================================================

This module implements PPO-based VM allocation optimization.
Supports two evaluation scenarios:
1. Minimize Resource Overload (CPU/RAM priority)
2. Optimize Operational Cost (cost/switching priority)
"""

from .config import PPOConfig, RewardConfig, SCENARIO_OVERLOAD, SCENARIO_COST
from .reward import compute_reward
from .environment import VMAllocationEnv
from .utils import (
    load_data,
    load_vm_catalog,
    generate_forecasts_for_rl,
    parse_allocation_string,
    evaluate_episode,
)

__all__ = [
    "PPOConfig",
    "RewardConfig",
    "SCENARIO_OVERLOAD",
    "SCENARIO_COST",
    "compute_reward",
    "VMAllocationEnv",
    "load_data",
    "load_vm_catalog",
    "generate_forecasts_for_rl",
    "parse_allocation_string",
    "evaluate_episode",
]


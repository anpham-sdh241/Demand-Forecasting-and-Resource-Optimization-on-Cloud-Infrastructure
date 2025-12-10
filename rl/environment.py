"""
Gymnasium Environment for VM Allocation
========================================

Implements VMAllocationEnv:
- State: forecast + current_vms + time features
- Action: VM counts for each VM type
- Reward: Based on SLA, overflow, switching cost, VM cost, efficiency
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import (
    RewardConfig,
    PPOConfig,
    HOST_SPEC,
    MAX_VMS_PER_TYPE,
    SCENARIO_OVERLOAD,
    SCENARIO_COST,
)
from .reward import compute_reward
from .utils import (
    load_data,
    load_vm_catalog,
    compute_resource_requirements,
    generate_forecasts_for_rl,
    get_time_features,
    compute_vm_resources,
    action_to_allocation,
)


class VMAllocationEnv(gym.Env):
    """
    Gymnasium environment for VM allocation using PPO.
    
    Observation Space:
        - Current demand: [cpu_required, mem_required, cpu_overflow, mem_overflow] (4)
        - Forecast horizon: [cpu, mem, load] × horizon steps (3 × horizon)
        - Current VMs: counts for each VM type (n_vm_types)
        - Time features: [hour_sin, hour_cos, dow_sin, dow_cos, is_peak] (5)
        
    Action Space:
        - Discrete count for each VM type: [0, MAX_VMS_PER_TYPE]
        - Using MultiDiscrete space
    
    Reward:
        - Scenario-dependent weighted combination of:
          SLA penalty, overflow penalty, switching cost, VM cost, efficiency bonus
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(
        self,
        scenario: str = SCENARIO_OVERLOAD,
        data: Optional[pd.DataFrame] = None,
        vm_catalog: Optional[Dict] = None,
        host_spec: Optional[Dict] = None,
        ppo_config: Optional[PPOConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        episode_length: Optional[int] = None,
        horizon: int = 12,
        max_vms_per_type: int = MAX_VMS_PER_TYPE,
        random_start: bool = True,
        render_mode: Optional[str] = None,
        forecast_cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize the VM Allocation Environment.
        
        Args:
            scenario: "overload" or "cost" - determines reward weights
            data: DataFrame with demand data. If None, loads from default path.
            vm_catalog: VM specifications. If None, loads from default path.
            host_spec: Host machine specs. If None, uses default.
            ppo_config: PPO configuration. If None, uses default.
            reward_config: Reward configuration. If None, derived from scenario.
            episode_length: Steps per episode. If None, uses config default.
            horizon: Forecast horizon steps.
            max_vms_per_type: Maximum VMs per type (action space bound).
            random_start: Whether to randomize episode start point.
            render_mode: Rendering mode (None or "human").
        """
        super().__init__()
        
        self.scenario = scenario
        self.render_mode = render_mode
        self.random_start = random_start
        self.horizon = horizon
        self.max_vms_per_type = max_vms_per_type
        self.forecast_cache = forecast_cache  # Optional precomputed forecasts per target
        
        # Load configurations
        self.ppo_config = ppo_config or PPOConfig()
        self.host_spec = host_spec or HOST_SPEC
        
        # Set reward config based on scenario
        if reward_config is not None:
            self.reward_config = reward_config
        elif scenario == SCENARIO_COST:
            self.reward_config = RewardConfig.cost_scenario()
        else:
            self.reward_config = RewardConfig.overload_scenario()
        
        # Episode length
        self.episode_length = episode_length or self.ppo_config.episode_length
        
        # Load data
        if data is not None:
            self.data = data
        else:
            train_df, _ = load_data()
            self.data = train_df
        
        # Load VM catalog
        self.vm_catalog = vm_catalog or load_vm_catalog()
        self.vm_types = list(self.vm_catalog.keys())
        self.n_vm_types = len(self.vm_types)
        
        # Compute thresholds
        self.cpu_threshold = (
            self.host_spec["total_cpu_cores"] * 
            self.host_spec["cpu_threshold_pct"] / 100.0
        )
        self.mem_threshold = (
            self.host_spec["total_memory_gb"] * 
            self.host_spec["memory_threshold_pct"] / 100.0
        )
        
        # Define observation space
        # [demand(4) + forecast(3*horizon) + current_vms(n_vm_types) + time_features(5)]
        obs_dim = 4 + (3 * self.horizon) + self.n_vm_types + 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Define action space (MultiDiscrete: count for each VM type)
        self.action_space = spaces.MultiDiscrete(
            [self.max_vms_per_type + 1] * self.n_vm_types
        )
        
        # State variables
        self.current_step = 0
        self.start_idx = 0
        self.current_vms: Dict[str, int] = {}
        self.prev_vms: Dict[str, int] = {}
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_infos = []
    
    def _get_obs(self) -> np.ndarray:
        """
        Construct observation vector from current state.
        
        Returns:
            Observation array
        """
        idx = self.start_idx + self.current_step
        
        # Handle index bounds
        if idx >= len(self.data):
            idx = len(self.data) - 1
        
        row = self.data.iloc[idx]
        
        # Current demand (4 features)
        cpu_req, mem_req, cpu_over, mem_over = compute_resource_requirements(
            row, self.host_spec
        )
        demand_features = np.array([cpu_req, mem_req, cpu_over, mem_over], dtype=np.float32)
        
        # Forecast features (3 × horizon)
        if self.forecast_cache:
            # Use cached model-based forecasts if available
            fc_list = []
            targets = ["cpu_total_usage", "memory_usage_pct", "system_load"]
            for tgt in targets:
                tgt_cache = self.forecast_cache.get(tgt)
                if tgt_cache is not None and idx < len(tgt_cache):
                    fc_vec = tgt_cache[idx]
                else:
                    # Fallback to zeros if out of range
                    fc_vec = np.zeros(self.horizon, dtype=np.float32)
                fc_list.append(fc_vec)
            forecasts = np.stack(fc_list, axis=1)  # shape (horizon, 3)
            forecast_features = forecasts.flatten()
        else:
            forecasts = generate_forecasts_for_rl(self.data, idx, self.horizon)
            forecast_features = forecasts.flatten()
        
        # Current VM counts (n_vm_types)
        vm_features = np.array(
            [self.current_vms.get(vm_type, 0) for vm_type in self.vm_types],
            dtype=np.float32,
        )
        
        # Time features (5)
        hour, dow, is_peak = get_time_features(row)
        # Encode cyclical features
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        time_features = np.array(
            [hour_sin, hour_cos, dow_sin, dow_cos, float(is_peak)],
            dtype=np.float32,
        )
        
        # Concatenate all features
        obs = np.concatenate([
            demand_features,
            forecast_features,
            vm_features,
            time_features,
        ])
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get info dictionary for current state.
        
        Returns:
            Info dictionary
        """
        idx = self.start_idx + self.current_step
        if idx >= len(self.data):
            idx = len(self.data) - 1
        
        row = self.data.iloc[idx]
        cpu_req, mem_req, cpu_over, mem_over = compute_resource_requirements(
            row, self.host_spec
        )
        cpu_alloc, mem_alloc, vm_cost = compute_vm_resources(
            self.current_vms, self.vm_catalog
        )
        
        return {
            "step": self.current_step,
            "data_idx": idx,
            "cpu_required": cpu_req,
            "mem_required": mem_req,
            "cpu_overflow": cpu_over,
            "mem_overflow": mem_over,
            "cpu_allocated": cpu_alloc,
            "mem_allocated": mem_alloc,
            "vm_cost": vm_cost,
            "current_vms": self.current_vms.copy(),
            "scenario": self.scenario,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Determine start index
        max_start = max(0, len(self.data) - self.episode_length - self.horizon)
        
        if self.random_start and max_start > 0:
            self.start_idx = self.np_random.integers(0, max_start)
        else:
            self.start_idx = 0
        
        # Reset state
        self.current_step = 0
        self.current_vms = {vm_type: 0 for vm_type in self.vm_types}
        self.prev_vms = {vm_type: 0 for vm_type in self.vm_types}
        
        # Reset tracking
        self.episode_rewards = []
        self.episode_infos = []
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Array of VM counts for each type
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to VM allocation
        self.prev_vms = self.current_vms.copy()
        self.current_vms = action_to_allocation(action, self.vm_types)
        
        # Get current demand
        idx = self.start_idx + self.current_step
        if idx >= len(self.data):
            idx = len(self.data) - 1
        
        row = self.data.iloc[idx]
        cpu_req, mem_req, _, _ = compute_resource_requirements(row, self.host_spec)
        
        # Compute allocated resources
        cpu_alloc, mem_alloc, vm_cost = compute_vm_resources(
            self.current_vms, self.vm_catalog
        )
        
        # Compute reward
        reward, reward_breakdown = compute_reward(
            cpu_required=cpu_req,
            mem_required=mem_req,
            cpu_threshold=self.cpu_threshold,
            mem_threshold=self.mem_threshold,
            cpu_allocated=cpu_alloc,
            mem_allocated=mem_alloc,
            vm_allocation=self.current_vms,
            prev_vm_allocation=self.prev_vms,
            vm_catalog=self.vm_catalog,
            config=self.reward_config,
        )
        
        # Track episode progress
        self.episode_rewards.append(reward)
        
        # Advance step
        self.current_step += 1
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        # Get new observation and info
        observation = self._get_obs()
        info = self._get_info()
        info.update(reward_breakdown)
        
        # Add episode summary on truncation
        if truncated:
            info["episode_reward"] = sum(self.episode_rewards)
            info["episode_length"] = self.current_step
        
        self.episode_infos.append(info)
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the current state."""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"\n=== Step {self.current_step} ===")
            print(f"Scenario: {self.scenario}")
            print(f"CPU Required: {info['cpu_required']:.2f} cores")
            print(f"Memory Required: {info['mem_required']:.2f} GB")
            print(f"CPU Allocated: {info['cpu_allocated']:.2f} cores")
            print(f"Memory Allocated: {info['mem_allocated']:.2f} GB")
            print(f"VM Cost/hour: ${info['vm_cost']:.4f}")
            print(f"Current VMs: {info['current_vms']}")
    
    def close(self):
        """Clean up resources."""
        pass


# Import pandas for type hints (deferred to avoid circular import)
import pandas as pd


def make_env(
    scenario: str = SCENARIO_OVERLOAD,
    is_training: bool = True,
    use_model_forecast: bool = True,
    **kwargs,
) -> VMAllocationEnv:
    """
    Factory function to create VMAllocationEnv.
    
    Args:
        scenario: "overload" or "cost"
        is_training: If True, use training data; else use test data
        **kwargs: Additional arguments for VMAllocationEnv
        
    Returns:
        Configured VMAllocationEnv instance
    """
    train_df, test_df = load_data()
    data = train_df if is_training else test_df

    forecast_cache = None
    if use_model_forecast and not is_training:
        try:
            from .utils import TARGET_MODEL_MAP, build_step_forecasts
            forecast_cache = build_step_forecasts(
                target_model_map=TARGET_MODEL_MAP,
                horizon=kwargs.get("horizon", 12),
                split="test",
            )
        except Exception as e:
            print(f"Warning: model-based forecasts unavailable, falling back to perfect forecasts. Details: {e}")
            forecast_cache = None
    
    return VMAllocationEnv(
        scenario=scenario,
        data=data,
        random_start=is_training,
        forecast_cache=forecast_cache,
        **kwargs,
    )


def make_vec_env(
    scenario: str = SCENARIO_OVERLOAD,
    n_envs: int = 4,
    is_training: bool = True,
    **kwargs,
):
    """
    Create vectorized environment for parallel training.
    
    Args:
        scenario: "overload" or "cost"
        n_envs: Number of parallel environments
        is_training: If True, use training data
        **kwargs: Additional arguments for VMAllocationEnv
        
    Returns:
        Vectorized environment
    """
    from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
    
    def env_fn():
        return make_env(scenario=scenario, is_training=is_training, **kwargs)
    
    return sb3_make_vec_env(env_fn, n_envs=n_envs)


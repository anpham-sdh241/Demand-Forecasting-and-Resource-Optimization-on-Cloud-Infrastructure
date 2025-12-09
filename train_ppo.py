"""
Train PPO Agent for VM Allocation
==================================

This script trains PPO agents for two scenarios:
1. Minimize Resource Overload (overload-first)
2. Optimize Operational Cost (cost-first)

Usage:
    python train_ppo.py --scenario overload --timesteps 500000
    python train_ppo.py --scenario cost --timesteps 500000
    python train_ppo.py --scenario both --timesteps 500000

Outputs:
    - Trained models: rl_models/ppo_overload.zip, rl_models/ppo_cost.zip
    - TensorBoard logs: tensorboard_logs/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.config import PPOConfig, SCENARIO_OVERLOAD, SCENARIO_COST
from rl.environment import make_env


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "rl_models"
TENSORBOARD_DIR = PROJECT_ROOT / "tensorboard_logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
TENSORBOARD_DIR.mkdir(exist_ok=True)


def create_callbacks(
    scenario: str,
    eval_env,
    save_freq: int = 10000,
) -> CallbackList:
    """
    Create training callbacks for logging and checkpointing.
    
    Args:
        scenario: Scenario name for file naming
        eval_env: Environment for evaluation
        save_freq: Checkpoint save frequency
        
    Returns:
        Combined callback list
    """
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(MODELS_DIR / f"checkpoints_{scenario}"),
        name_prefix=f"ppo_{scenario}",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / f"best_{scenario}"),
        log_path=str(MODELS_DIR / f"eval_logs_{scenario}"),
        eval_freq=save_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    return CallbackList([checkpoint_callback, eval_callback])


def train_ppo(
    scenario: str,
    config: PPOConfig,
    total_timesteps: int | None = None,
    n_envs: int = 4,
    continue_training: bool = False,
) -> PPO:
    """
    Train a PPO agent for the specified scenario.
    
    Args:
        scenario: "overload" or "cost"
        config: PPO configuration
        total_timesteps: Override total training timesteps
        n_envs: Number of parallel environments
        continue_training: Whether to continue from existing model
        
    Returns:
        Trained PPO model
    """
    print(f"\n{'='*60}")
    print(f"Training PPO for scenario: {scenario.upper()}")
    print(f"{'='*60}")
    
    timesteps = total_timesteps or config.total_timesteps
    
    # Create training environments
    def make_train_env():
        env = make_env(scenario=scenario, is_training=True)
        env = Monitor(env)
        return env
    
    train_envs = DummyVecEnv([make_train_env for _ in range(n_envs)])
    train_envs = VecNormalize(train_envs, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    def make_eval_env():
        env = make_env(scenario=scenario, is_training=False)
        env = Monitor(env)
        return env
    
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Create or load model
    model_path = MODELS_DIR / f"ppo_{scenario}.zip"
    
    if continue_training and model_path.exists():
        print(f"Loading existing model from {model_path}")
        model = PPO.load(
            str(model_path),
            env=train_envs,
            tensorboard_log=str(TENSORBOARD_DIR),
        )
    else:
        print("Creating new PPO model")
        model = PPO(
            policy="MlpPolicy",
            env=train_envs,
            verbose=1,
            **config.to_sb3_kwargs(),
        )
    
    # Create callbacks
    callbacks = create_callbacks(
        scenario=scenario,
        eval_env=eval_env,
        save_freq=config.save_freq,
    )
    
    # Train
    print(f"\nStarting training for {timesteps:,} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not continue_training,
    )
    
    training_time = datetime.now() - start_time
    print(f"\nTraining completed in {training_time}")
    
    # Save final model
    final_model_path = MODELS_DIR / f"ppo_{scenario}.zip"
    model.save(str(final_model_path))
    print(f"Model saved to {final_model_path}")
    
    # Save VecNormalize statistics
    vec_normalize_path = MODELS_DIR / f"ppo_{scenario}_vecnormalize.pkl"
    train_envs.save(str(vec_normalize_path))
    print(f"VecNormalize saved to {vec_normalize_path}")
    
    # Clean up
    train_envs.close()
    eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent for VM Allocation"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["overload", "cost", "both"],
        default="both",
        help="Scenario to train: 'overload', 'cost', or 'both'",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--continue",
        dest="continue_training",
        action="store_true",
        help="Continue training from existing model",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Override episode length",
    )
    
    args = parser.parse_args()
    
    # Build config
    config = PPOConfig()
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.episode_length:
        config.episode_length = args.episode_length
    
    print("="*60)
    print("PPO Training for VM Allocation")
    print("="*60)
    print(f"Scenario(s): {args.scenario}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Episode length: {config.episode_length}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Continue training: {args.continue_training}")
    print("="*60)
    
    scenarios = []
    if args.scenario == "both":
        scenarios = [SCENARIO_OVERLOAD, SCENARIO_COST]
    else:
        scenarios = [args.scenario]
    
    for scenario in scenarios:
        train_ppo(
            scenario=scenario,
            config=config,
            total_timesteps=config.total_timesteps,
            n_envs=args.n_envs,
            continue_training=args.continue_training,
        )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Models saved in: {MODELS_DIR}")
    print(f"TensorBoard logs: {TENSORBOARD_DIR}")
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir {TENSORBOARD_DIR}")


if __name__ == "__main__":
    main()


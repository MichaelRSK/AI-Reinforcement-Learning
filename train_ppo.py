"""
PPO Training Loop for Pong

This module implements Proximal Policy Optimization (PPO) for training
a policy network to play Pong using Gymnasium.

PPO Key Features:
1. Actor-Critic architecture (policy + value function)
2. Clipped surrogate objective (prevents destructive updates)
3. Multiple epochs per batch (sample efficiency)
4. Generalized Advantage Estimation (variance reduction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import ale_py
import os
import json
from datetime import datetime
from collections import deque

from ppo_network import PPOActorCritic
from preprocessing import preprocess_frame


class PPOTrainer:
    """
    PPO trainer for Pong environment.
    
    Implements PPO algorithm with:
    - Clipped surrogate objective
    - Value function learning
    - Generalized Advantage Estimation (GAE)
    - Multiple epochs per rollout
    """
    
    def __init__(
        self,
        env_name="ALE/Pong-v5",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        batch_size=64,
        checkpoint_dir="checkpoints_ppo",
        device=None
    ):
        """
        Initialize the PPO trainer.
        
        Args:
            env_name: Gymnasium environment name
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            gae_lambda: Lambda for GAE calculation
            clip_epsilon: PPO clipping parameter (epsilon)
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Max gradient norm for clipping
            ppo_epochs: Number of optimization epochs per rollout
            batch_size: Minibatch size for PPO updates
            checkpoint_dir: Directory to save model checkpoints
            device: PyTorch device (cuda/cpu), auto-detected if None
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Environment setup
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.num_actions = self.env.action_space.n
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Create Actor-Critic network
        self.policy_net = PPOActorCritic(input_channels=1, num_actions=self.num_actions)
        self.policy_net.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.reward_stats_history = []
        self.training_stats = {
            "episode": [],
            "reward": [],
            "length": [],
            "avg_reward": [],
            "avg_length": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": []
        }
        
        # Frame tracking for preprocessing
        self.previous_frame = None
        
    def preprocess_observation(self, observation):
        """
        Preprocess a single observation frame.
        
        Args:
            observation: Raw RGB frame from environment
            
        Returns:
            Preprocessed frame (80x80 grayscale, normalized)
        """
        processed = preprocess_frame(
            observation,
            previous_frame=self.previous_frame,
            target_size=(80, 80),
            normalize=True
        )
        # Update previous frame for next differencing
        self.previous_frame = observation.copy()
        return processed
    
    def collect_rollout(self, num_steps=2048):
        """
        Collect a rollout of experiences.
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            values: List of value estimates
            log_probs: List of action log probabilities
            dones: List of done flags
        """
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        # Reset if needed
        if not hasattr(self, 'current_obs'):
            observation, info = self.env.reset()
            self.previous_frame = None
            self.current_obs = self.preprocess_observation(observation)
        
        for _ in range(num_steps):
            # Get action from policy
            action, log_prob, value = self.policy_net.get_action(
                self.current_obs, 
                deterministic=False
            )
            
            # Take step in environment
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            states.append(self.current_obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)
            
            # Update state
            if not done:
                self.current_obs = self.preprocess_observation(next_observation)
            else:
                # Reset for next episode
                observation, info = self.env.reset()
                self.previous_frame = None
                self.current_obs = self.preprocess_observation(observation)
        
        return states, actions, rewards, values, log_probs, dones
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE balances bias and variance in advantage estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for the next state
            
        Returns:
            advantages: List of advantage estimates
            returns: List of returns (for value function training)
        """
        advantages = []
        returns = []
        
        gae = 0
        next_val = next_value
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_val = 0
                gae = 0
            
            # TD error: r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * next_val - values[t]
            
            # GAE: delta + gamma * lambda * GAE
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
            next_val = values[t]
        
        return advantages, returns
    
    def ppo_update(self, states, actions, old_log_probs, advantages, returns):
        """
        Perform PPO update using collected experiences.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Log probabilities under old policy
            advantages: Advantage estimates
            returns: Return estimates (targets for value function)
            
        Returns:
            policy_loss: Policy loss value
            value_loss: Value loss value
            entropy: Entropy value
        """
        # Convert to tensors
        states_tensor = torch.stack([torch.from_numpy(s).float() for s in states]).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Multiple epochs of optimization
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            # Mini-batch updates
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Evaluate actions under current policy
                action_log_probs, state_values, entropy = self.policy_net.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Ratio for PPO
                ratio = torch.exp(action_log_probs - batch_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = F.mse_loss(state_values, batch_returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.value_loss_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Return average losses
        return (
            total_policy_loss / num_updates,
            total_value_loss / num_updates,
            total_entropy / num_updates
        )
    
    def train_step(self, rollout_steps=2048):
        """
        Perform one training step (collect rollout + PPO update).
        
        Args:
            rollout_steps: Number of steps to collect before update
            
        Returns:
            episode_rewards: List of episode rewards collected
            policy_loss: Policy loss value
            value_loss: Value loss value
            entropy: Entropy value
        """
        # Collect rollout
        states, actions, rewards, values, log_probs, dones = self.collect_rollout(rollout_steps)
        
        # Get next value for GAE
        with torch.no_grad():
            _, next_value, _ = self.policy_net.get_action(self.current_obs)
            next_value = next_value.item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # PPO update
        policy_loss, value_loss, entropy = self.ppo_update(
            states, actions, log_probs, advantages, returns
        )
        
        # Calculate episode rewards (for logging)
        episode_rewards_list = []
        current_reward = 0
        for r, done in zip(rewards, dones):
            current_reward += r
            if done:
                episode_rewards_list.append(current_reward)
                current_reward = 0
        
        return episode_rewards_list, policy_loss, value_loss, entropy
    
    def save_checkpoint(self, episode, filename=None):
        """
        Save model checkpoint.
        
        Args:
            episode: Current episode number
            filename: Optional filename, auto-generated if None
        """
        if filename is None:
            filename = f"ppo_checkpoint_episode_{episode}.pt"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            "episode": episode,
            "model_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "value_loss_coef": self.value_loss_coef,
                "entropy_coef": self.entropy_coef
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint.get("training_stats", self.training_stats)
        
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint["episode"]
    
    def train(self, num_episodes=1000, rollout_steps=2048, save_frequency=100, print_frequency=10):
        """
        Main training loop.
        
        Args:
            num_episodes: Target number of episodes to train
            rollout_steps: Steps per rollout before PPO update
            save_frequency: Save checkpoint every N episodes
            print_frequency: Print statistics every N episodes
        """
        print("\n" + "=" * 70)
        print("  🚀 STARTING PPO TRAINING FOR PONG")
        print("=" * 70)
        print(f"  Environment:          {self.env_name}")
        print(f"  Device:               {self.device}")
        print(f"  Action Space:         {self.num_actions} actions")
        print(f"  Learning Rate:        {self.learning_rate}")
        print(f"  Discount Factor:      {self.gamma}")
        print(f"  GAE Lambda:           {self.gae_lambda}")
        print(f"  Clip Epsilon:         {self.clip_epsilon}")
        print(f"  Target Episodes:      {num_episodes}")
        print(f"  Rollout Steps:        {rollout_steps}")
        print(f"  PPO Epochs:           {self.ppo_epochs}")
        print(f"  Batch Size:           {self.batch_size}")
        print("=" * 70)
        print(f"  PPO Features:")
        print(f"    • Actor-Critic architecture")
        print(f"    • Clipped surrogate objective")
        print(f"    • Generalized Advantage Estimation")
        print(f"    • Multiple optimization epochs")
        print("=" * 70 + "\n")
        
        episode_count = 0
        step_count = 0
        
        while episode_count < num_episodes:
            # Train step
            episode_rewards_list, policy_loss, value_loss, entropy = self.train_step(rollout_steps)
            
            # Update episode count
            episode_count += len(episode_rewards_list)
            step_count += rollout_steps
            
            # Store statistics for all completed episodes
            for ep_reward in episode_rewards_list:
                self.episode_rewards.append(ep_reward)
                self.training_stats["episode"].append(len(self.episode_rewards))
                self.training_stats["reward"].append(ep_reward)
                self.training_stats["policy_loss"].append(policy_loss)
                self.training_stats["value_loss"].append(value_loss)
                self.training_stats["entropy"].append(entropy)
            
            # Print progress (only if episodes were completed this step)
            if len(episode_rewards_list) > 0 and len(self.episode_rewards) % print_frequency == 0:
                ep_num = len(self.episode_rewards)
                recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                avg_reward = np.mean(recent_rewards)
                
                print(f"\n{'='*70}")
                print(f"Episode {ep_num}/{num_episodes} ({ep_num/num_episodes*100:.1f}% complete)")
                print(f"{'='*70}")
                print(f"  Recent Reward (10):   {avg_reward:+.2f}")
                print(f"  Policy Loss:          {policy_loss:.4f}")
                print(f"  Value Loss:           {value_loss:.4f}")
                print(f"  Entropy:              {entropy:.4f}")
                print(f"  Total Steps:          {step_count:,}")
                
                # Show trend
                if len(self.episode_rewards) >= 20:
                    recent_avg = np.mean(self.episode_rewards[-10:])
                    prev_avg = np.mean(self.episode_rewards[-20:-10])
                    if recent_avg > prev_avg + 0.5:
                        trend = "📈 IMPROVING"
                    elif recent_avg < prev_avg - 0.5:
                        trend = "📉 DECLINING"
                    else:
                        trend = "➡️  STABLE"
                    print(f"  Trend:                {trend}")
                print(f"{'='*70}")
            
            # Save checkpoint (only if episodes were completed this step)
            if len(episode_rewards_list) > 0 and len(self.episode_rewards) % save_frequency == 0:
                self.save_checkpoint(len(self.episode_rewards))
        
        # Save final checkpoint
        self.save_checkpoint(len(self.episode_rewards), "ppo_final_checkpoint.pt")
        
        print("\n" + "=" * 70)
        print("  🎉 PPO TRAINING COMPLETE! 🎉")
        print("=" * 70)
        print(f"  Total Episodes:           {len(self.episode_rewards)}")
        print(f"  Average Reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"  Best Episode Reward:      {max(self.episode_rewards):.2f}")
        print(f"  Checkpoints saved in:     {self.checkpoint_dir}")
        print("=" * 70)
        
        # Save training statistics to JSON
        stats_file = os.path.join(self.checkpoint_dir, "ppo_training_stats.json")
        with open(stats_file, "w") as f:
            json.dump(self.training_stats, f, indent=2)
        print(f"Training statistics saved to: {stats_file}\n")
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main():
    """Main function to run PPO training."""
    # Create trainer
    trainer = PPOTrainer(
        env_name="ALE/Pong-v5",
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        checkpoint_dir="checkpoints_ppo"
    )
    
    try:
        # Train for specified number of episodes
        trainer.train(
            num_episodes=1000,
            rollout_steps=2048,
            save_frequency=100,
            print_frequency=10
        )
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

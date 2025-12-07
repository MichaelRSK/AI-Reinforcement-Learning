"""
REINFORCE Training Loop for Pong

This module implements the REINFORCE (Policy Gradient) algorithm for training
a policy network to play Pong using Gymnasium.

Algorithm:
1. Collect episodes by interacting with the environment
2. Compute discounted returns (rewards)
3. Normalize returns (baseline subtraction)
4. Compute policy gradient loss
5. Update network parameters
6. Save checkpoints periodically
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import ale_py
import os
import json
from datetime import datetime
from collections import deque

from policy_network import PolicyNetwork
from preprocessing import preprocess_frame


class REINFORCETrainer:
    """
    REINFORCE trainer for Pong environment.
    
    Implements the REINFORCE algorithm:
    - Collects episodes of (state, action, reward)
    - Computes discounted returns
    - Normalizes returns (optional baseline)
    - Updates policy using policy gradient
    """
    
    def __init__(
        self,
        env_name="ALE/Pong-v5",
        learning_rate=1e-4,
        gamma=0.99,
        normalize_returns=True,
        checkpoint_dir="checkpoints",
        device=None
    ):
        """
        Initialize the REINFORCE trainer.
        
        Args:
            env_name: Gymnasium environment name
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            normalize_returns: Whether to normalize returns (subtract mean)
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
        self.normalize_returns = normalize_returns
        
        # Create policy network
        self.policy_net = PolicyNetwork(input_channels=1, num_actions=self.num_actions)
        self.policy_net.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.reward_stats_history = []  # Track detailed reward statistics
        self.training_stats = {
            "episode": [],
            "reward": [],
            "length": [],
            "avg_reward": [],
            "avg_length": []
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
    
    def compute_shaped_reward(self, observation, previous_observation, env_reward, action):
        """
        Compute shaped reward based on game state.
        
        CURRENTLY DISABLED: Reward shaping is turned off to establish a clean baseline.
        The agent will learn purely from environment rewards (±1 for scoring).
        
        Previous shaping had detection bugs:
        - Ball detection was picking up paddles, scores, and center line
        - "Ball on agent's side" was always true (detected own paddle)
        - Alignment reward was checking if paddle aligned with itself
        
        These bugs caused shaped bonuses of +10-16 per episode, drowning out
        the true objective. Once baseline learning is confirmed, we can add
        back PROPER reward shaping with correct ball/paddle isolation.
        
        Args:
            observation: Current raw RGB frame (210, 160, 3)
            previous_observation: Previous raw RGB frame (None if first frame)
            env_reward: Original environment reward (-1, 0, or +1)
            action: Action taken (0-5)
            
        Returns:
            shaped_reward: Environment reward only (no shaping applied)
        """
        # Return pure environment reward for now
        shaped_reward = float(env_reward)
        
        # TODO: Add back proper reward shaping once baseline is established:
        # 1. Isolate ball position (small white square, 2-6 pixels, moving)
        # 2. Track ball proximity only when ball is actually detected
        # 3. Reward successful hits (ball moving away from agent)
        # 4. Keep shaping amounts 10-100x smaller than environment rewards
        
        return shaped_reward
    
    def collect_episode(self, max_steps=10000):
        """
        Collect one episode of experience with reward shaping.
        
        Reward shaping compares consecutive frames to detect ball position
        and paddle hits for better learning signal.
        
        Args:
            max_steps: Maximum steps per episode (safety limit)
            
        Returns:
            states: List of preprocessed states
            actions: List of actions taken
            rewards: List of rewards received (with shaping)
            log_probs: List of log probabilities of actions
            env_rewards: List of original environment rewards (for statistics)
        """
        # Reset environment
        observation, info = self.env.reset()
        self.previous_frame = None  # Reset frame differencing
        
        # Preprocess initial observation
        state = self.preprocess_observation(observation)
        
        # Storage for episode
        states = []
        actions = []
        rewards = []
        log_probs = []
        env_rewards = []  # Track original environment rewards separately
        
        # Run episode
        step_count = 0
        done = False
        
        while not done and step_count < max_steps:
            # Get action from policy
            action, log_prob = self.policy_net.get_action(state, deterministic=False)
            
            # Take step in environment
            next_observation, env_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Apply reward shaping
            shaped_reward = self.compute_shaped_reward(
                next_observation, 
                observation,  # Compare with current frame, not previous
                env_reward, 
                action
            )
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(shaped_reward)  # Use shaped reward instead of env_reward
            env_rewards.append(env_reward)  # Track original env reward for statistics
            log_probs.append(log_prob)
            
            # Update state
            if not done:
                state = self.preprocess_observation(next_observation)
            
            observation = next_observation  # Update for next iteration
            step_count += 1
        
        return states, actions, rewards, log_probs, env_rewards
    
    def compute_discounted_returns(self, rewards):
        """
        Compute discounted returns (cumulative future rewards).
        
        Args:
            rewards: List of rewards from an episode
            
        Returns:
            returns: List of discounted returns
        """
        returns = []
        G = 0  # Cumulative return
        
        # Compute returns backwards (from end of episode)
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)  # Insert at beginning to maintain order
        
        return returns
    
    def _normalize_returns_array(self, returns):
        """
        Normalize returns by subtracting mean and dividing by std.
        This reduces variance in policy gradient updates.
        
        Args:
            returns: List of returns
            
        Returns:
            normalized_returns: Normalized returns
        """
        returns_array = np.array(returns)
        mean = returns_array.mean()
        std = returns_array.std()
        
        # Avoid division by zero
        if std < 1e-8:
            return returns_array - mean
        
        normalized = (returns_array - mean) / std
        return normalized
    
    def update_policy(self, states, actions, returns):
        """
        Update policy network using REINFORCE algorithm.
        
        Args:
            states: List of preprocessed states
            actions: List of actions taken
            returns: List of discounted returns (normalized if applicable)
            
        Returns:
            loss: Policy gradient loss value
        """
        # Convert to tensors
        states_tensor = torch.stack([torch.from_numpy(s).float() for s in states]).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Forward pass to get log probabilities
        _, log_probs = self.policy_net(states_tensor)
        
        # Get log probabilities for the actions actually taken
        action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Policy gradient loss: -log_prob * return
        # We negate because we want to maximize, but optimizer minimizes
        loss = -(action_log_probs * returns_tensor).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self):
        """
        Train on one episode.
        
        Returns:
            episode_reward: Total reward for the episode
            episode_length: Length of the episode
            loss: Policy gradient loss
            reward_stats: Dictionary with detailed reward statistics
        """
        # Collect episode
        states, actions, rewards, log_probs, env_rewards = self.collect_episode()
        
        # Compute statistics
        episode_reward = sum(rewards)
        episode_length = len(rewards)
        
        # Analyze reward statistics
        env_rewards_array = np.array(env_rewards)
        points_scored = np.sum(env_rewards_array == 1)  # Agent scored
        points_lost = np.sum(env_rewards_array == -1)  # Opponent scored
        shaped_bonus = episode_reward - sum(env_rewards)  # Total shaping bonus
        
        reward_stats = {
            "points_scored": points_scored,
            "points_lost": points_lost,
            "shaped_bonus": shaped_bonus,
            "env_reward_total": sum(env_rewards)
        }
        
        # Compute discounted returns
        returns = self.compute_discounted_returns(rewards)
        
        # Normalize returns (optional, but recommended)
        if self.normalize_returns:
            returns = self._normalize_returns_array(returns)
        
        # Update policy
        loss = self.update_policy(states, actions, returns)
        
        return episode_reward, episode_length, loss, reward_stats
    
    def save_checkpoint(self, episode, filename=None):
        """
        Save model checkpoint.
        
        Args:
            episode: Current episode number
            filename: Optional filename, auto-generated if None
        """
        if filename is None:
            filename = f"checkpoint_episode_{episode}.pt"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            "episode": episode,
            "model_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "normalize_returns": self.normalize_returns
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
    
    def train(self, num_episodes=1000, save_frequency=100, print_frequency=10):
        """
        Main training loop.
        
        Args:
            num_episodes: Number of episodes to train
            save_frequency: Save checkpoint every N episodes
            print_frequency: Print statistics every N episodes
        """
        print("\n" + "=" * 70)
        print("  🎮 STARTING REINFORCE TRAINING FOR PONG")
        print("=" * 70)
        print(f"  Environment:          {self.env_name}")
        print(f"  Device:               {self.device}")
        print(f"  Action Space:         {self.num_actions} actions")
        print(f"  Learning Rate:        {self.learning_rate}")
        print(f"  Discount Factor:      {self.gamma}")
        print(f"  Normalize Returns:    {self.normalize_returns}")
        print(f"  Total Episodes:       {num_episodes}")
        print(f"  Save Every:           {save_frequency} episodes")
        print(f"  Print Every:          {print_frequency} episodes")
        print("=" * 70)
        print(f"  Reward Shaping:       🚫 DISABLED (Pure Environment Rewards)")
        print(f"                        Agent learns from ±1 scoring only")
        print("=" * 70 + "\n")
        
        # Training loop
        for episode in range(1, num_episodes + 1):
            # Train on one episode
            episode_reward, episode_length, loss, reward_stats = self.train_episode()
            
            # Store statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.reward_stats_history.append(reward_stats)
            
            # Update training stats
            self.training_stats["episode"].append(episode)
            self.training_stats["reward"].append(episode_reward)
            self.training_stats["length"].append(episode_length)
            
            # Compute running averages
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
            else:
                avg_reward = episode_reward
                avg_length = episode_length
            
            self.training_stats["avg_reward"].append(avg_reward)
            self.training_stats["avg_length"].append(avg_length)
            
            # Print statistics
            if episode % print_frequency == 0:
                # Get reward statistics
                agent_score = reward_stats['points_scored']
                opponent_score = reward_stats['points_lost']
                env_reward = reward_stats['env_reward_total']
                shaped_bonus = reward_stats['shaped_bonus']
                
                # Print comprehensive but clean summary
                print(f"\n{'='*70}")
                print(f"Episode {episode}/{num_episodes} ({episode/num_episodes*100:.1f}% complete)")
                print(f"{'='*70}")
                print(f"  Game Score:        {int(agent_score):2d} - {int(opponent_score):2d}  (Agent - Opponent)")
                print(f"  Episode Length:    {episode_length:4d} steps")
                print(f"  Environment Reward: {env_reward:+6.1f}")
                print(f"  Shaped Bonus:       {shaped_bonus:+6.2f}")
                print(f"  Total Reward:       {episode_reward:+6.2f}")
                print(f"  Avg Reward (10):    {avg_reward:+6.2f}")
                print(f"  Loss:               {loss:8.4f}")
                
                # Show trend indicator
                if len(self.episode_rewards) >= 20:
                    recent_avg = np.mean(self.episode_rewards[-10:])
                    prev_avg = np.mean(self.episode_rewards[-20:-10])
                    if recent_avg > prev_avg + 0.5:
                        trend = "📈 IMPROVING"
                    elif recent_avg < prev_avg - 0.5:
                        trend = "📉 DECLINING"
                    else:
                        trend = "➡️  STABLE"
                    print(f"  Trend:              {trend}")
                print(f"{'='*70}")
            
            # Save checkpoint and print extended summary
            if episode % save_frequency == 0:
                self.save_checkpoint(episode)
                
                # Print extended summary every save_frequency episodes
                last_100_rewards = self.episode_rewards[-min(100, len(self.episode_rewards)):]
                last_100_stats = self.reward_stats_history[-min(100, len(self.reward_stats_history)):]
                
                total_points_scored = sum(s['points_scored'] for s in last_100_stats)
                total_points_lost = sum(s['points_lost'] for s in last_100_stats)
                avg_100_reward = np.mean(last_100_rewards)
                best_100_reward = max(last_100_rewards)
                
                print(f"\n{'#'*70}")
                print(f"  CHECKPOINT #{episode//save_frequency} - Episode {episode}")
                print(f"{'#'*70}")
                print(f"  Last {len(last_100_rewards)} Episodes Summary:")
                print(f"    Total Points Scored:  {int(total_points_scored)}")
                print(f"    Total Points Lost:    {int(total_points_lost)}")
                print(f"    Win Rate:             {total_points_scored/(total_points_scored + total_points_lost)*100:.1f}%")
                print(f"    Average Reward:       {avg_100_reward:+.2f}")
                print(f"    Best Reward:          {best_100_reward:+.2f}")
                print(f"{'#'*70}\n")
        
        # Save final checkpoint
        self.save_checkpoint(num_episodes, "final_checkpoint.pt")
        
        # Calculate final statistics
        last_100_stats = self.reward_stats_history[-min(100, len(self.reward_stats_history)):]
        total_points_scored = sum(s['points_scored'] for s in last_100_stats)
        total_points_lost = sum(s['points_lost'] for s in last_100_stats)
        
        print("\n" + "=" * 70)
        print("  🎉 TRAINING COMPLETE! 🎉")
        print("=" * 70)
        print(f"  Total Episodes:           {num_episodes}")
        print(f"  Total Training Steps:     {sum(self.episode_lengths):,}")
        print(f"\n  Performance (Last 100 Episodes):")
        print(f"    Average Reward:         {np.mean(self.episode_rewards[-100:]):+.2f}")
        print(f"    Best Episode Reward:    {max(self.episode_rewards[-100:]):+.2f}")
        print(f"    Points Scored:          {int(total_points_scored)}")
        print(f"    Points Lost:            {int(total_points_lost)}")
        print(f"    Win Rate:               {total_points_scored/(total_points_scored + total_points_lost)*100:.1f}%")
        print(f"\n  All-Time Best:")
        print(f"    Best Episode Reward:    {max(self.episode_rewards):+.2f}")
        print(f"\n  Files Saved:")
        print(f"    Checkpoints:            {self.checkpoint_dir}/")
        print(f"    Final Model:            final_checkpoint.pt")
        
        # Save training statistics to JSON
        stats_file = os.path.join(self.checkpoint_dir, "training_stats.json")
        with open(stats_file, "w") as f:
            json.dump(self.training_stats, f, indent=2)
        print(f"    Training Stats:         training_stats.json")
        print("=" * 70 + "\n")
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main():
    """Main function to run training."""
    # Create trainer
    trainer = REINFORCETrainer(
        env_name="ALE/Pong-v5",
        learning_rate=3e-4,  # Increased from 1e-4 for faster learning
        gamma=0.99,
        normalize_returns=True,
        checkpoint_dir="checkpoints"
    )
    
    try:
        # Train for specified number of episodes
        trainer.train(
            num_episodes=1000,
            save_frequency=100,
            print_frequency=10
        )
    finally:
        trainer.close()


if __name__ == "__main__":
    main()


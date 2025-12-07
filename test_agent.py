"""
Test and Evaluate Trained Agent

This script loads a trained policy network checkpoint and evaluates
its performance by running it in the Pong environment.
Supports both REINFORCE and PPO checkpoints.

Usage:
    # REINFORCE agent
    python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --render
    
    # PPO agent
    python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --render
    
    # Multiple episodes
    python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --episodes 10 --render
"""

import argparse
import torch
import numpy as np
import gymnasium as gym
import ale_py
import os
import sys

from train import REINFORCETrainer
from train_ppo import PPOTrainer
from preprocessing import preprocess_frame


def detect_checkpoint_type(checkpoint_path):
    """
    Detect whether checkpoint is REINFORCE or PPO based on path and contents.
    
    Returns:
        'reinforce' or 'ppo'
    """
    # Check path for hints
    if 'ppo' in checkpoint_path.lower() or 'checkpoints_ppo' in checkpoint_path:
        return 'ppo'
    
    # Try to load and check structure
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # PPO checkpoints have 'policy_net' key, REINFORCE has 'policy_net' too
        # But PPO might have different structure - check for 'optimizer' vs 'policy_optimizer'
        if 'policy_optimizer' in checkpoint or 'value_optimizer' in checkpoint:
            return 'ppo'
        # Check if it's a PPO trainer checkpoint
        if 'policy_net' in checkpoint and 'value_net' not in checkpoint:
            # Could be either, but if it's in ppo directory, assume PPO
            if 'ppo' in os.path.dirname(checkpoint_path).lower():
                return 'ppo'
        return 'reinforce'
    except:
        # Default to REINFORCE if can't determine
        return 'reinforce'


def test_agent(
    checkpoint_path,
    num_episodes=5,
    render=False,
    save_video=None,
    max_steps_per_episode=10000,
    agent_type=None
):
    """
    Test a trained agent by running it in the environment.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_episodes: Number of episodes to test
        render: Whether to render the environment visually
        save_video: Optional path to save video (e.g., "output.mp4")
        max_steps_per_episode: Maximum steps per episode (safety limit)
        agent_type: 'reinforce' or 'ppo' (auto-detected if None)
    
    Returns:
        Dictionary with test statistics
    """
    print("=" * 70)
    print("TESTING TRAINED AGENT")
    print("=" * 70)
    
    # Detect agent type if not specified
    if agent_type is None:
        agent_type = detect_checkpoint_type(checkpoint_path)
    
    print(f"Detected agent type: {agent_type.upper()}")
    
    # Create appropriate trainer
    if agent_type == 'ppo':
        trainer = PPOTrainer()
    else:
        trainer = REINFORCETrainer()
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = os.path.dirname(checkpoint_path) or "checkpoints"
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith(".pt"):
                    print(f"  - {os.path.join(checkpoint_dir, f)}")
        sys.exit(1)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        episode_num = trainer.load_checkpoint(checkpoint_path)
        print(f"Resuming from episode: {episode_num}")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        print(f"Trying to load as {agent_type}...")
        sys.exit(1)
    
    print("=" * 70)
    print()
    
    # Set to evaluation mode
    if agent_type == 'ppo':
        trainer.policy_net.eval()
    else:
        trainer.policy_net.eval()
    
    # Create environment for testing (with rendering if requested)
    if render or save_video:
        render_mode = "human" if render else "rgb_array"
        test_env = gym.make("ALE/Pong-v5", render_mode=render_mode)
    else:
        test_env = gym.make("ALE/Pong-v5")
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    wins = 0  # Episodes with positive reward
    
    print(f"Running {num_episodes} test episode(s)...")
    if render:
        print("(Close the render window or press Ctrl+C to stop early)")
    print()
    
    try:
        for episode in range(1, num_episodes + 1):
            # Reset environment
            observation, info = test_env.reset()
            previous_frame = None
            
            # Preprocess initial observation
            state = preprocess_frame(
                observation,
                previous_frame=previous_frame,
                target_size=(80, 80),
                normalize=True
            )
            previous_frame = observation.copy()
            
            # Run episode
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done and step_count < max_steps_per_episode:
                # Get action from policy (deterministic - use best action)
                if agent_type == 'ppo':
                    action, _, _ = trainer.policy_net.get_action(state, deterministic=True)
                else:
                    action, _ = trainer.policy_net.get_action(state, deterministic=True)
                
                # Take step
                next_observation, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                
                # Update statistics
                episode_reward += reward
                step_count += 1
                
                # Update state
                if not done:
                    state = preprocess_frame(
                        next_observation,
                        previous_frame=previous_frame,
                        target_size=(80, 80),
                        normalize=True
                    )
                    previous_frame = next_observation.copy()
            
            # Store statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            if episode_reward > 0:
                wins += 1
            
            # Print episode result
            print(f"Episode {episode}: Reward = {episode_reward:6.2f}, Length = {step_count}")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    
    finally:
        test_env.close()
    
    # Compute statistics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    best_reward = max(episode_rewards) if episode_rewards else 0
    worst_reward = min(episode_rewards) if episode_rewards else 0
    win_rate = (wins / len(episode_rewards) * 100) if episode_rewards else 0
    
    # Print results
    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.1f}")
    print(f"Best Episode Reward: {best_reward:.2f}")
    print(f"Worst Episode Reward: {worst_reward:.2f}")
    print(f"Win Rate: {win_rate:.1f}% (episodes with positive reward)")
    print("=" * 70)
    
    # Return statistics
    stats = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "best_reward": best_reward,
        "worst_reward": worst_reward,
        "win_rate": win_rate,
        "num_episodes": len(episode_rewards)
    }
    
    return stats


def compare_checkpoints(checkpoint_paths, num_episodes=5):
    """
    Compare performance of multiple checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        num_episodes: Number of episodes to test each checkpoint
    """
    print("=" * 70)
    print("COMPARING CHECKPOINTS")
    print("=" * 70)
    print()
    
    results = []
    
    for checkpoint_path in checkpoint_paths:
        print(f"Testing: {checkpoint_path}")
        agent_type = detect_checkpoint_type(checkpoint_path)
        stats = test_agent(checkpoint_path, num_episodes=num_episodes, render=False, agent_type=agent_type)
        results.append({
            "checkpoint": checkpoint_path,
            "avg_reward": stats["avg_reward"],
            "win_rate": stats["win_rate"],
            "agent_type": agent_type
        })
        print()
    
    # Print comparison
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Checkpoint':<40} {'Type':<10} {'Avg Reward':<15} {'Win Rate':<10}")
    print("-" * 70)
    for result in results:
        checkpoint_name = os.path.basename(result["checkpoint"])
        agent_type = result.get("agent_type", "unknown").upper()
        print(f"{checkpoint_name:<40} {agent_type:<10} {result['avg_reward']:>10.2f}     {result['win_rate']:>6.1f}%")
    print("=" * 70)


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test a trained Pong agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # REINFORCE agent - basic test
  python test_agent.py --checkpoint checkpoints/final_checkpoint.pt
  
  # REINFORCE agent - with visual rendering
  python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --render
  
  # PPO agent - with visual rendering
  python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --render
  
  # PPO agent - multiple episodes
  python test_agent.py --checkpoint checkpoints_ppo/ppo_final_checkpoint.pt --episodes 10 --render
  
  # Compare multiple checkpoints
  python test_agent.py --compare checkpoints/checkpoint_episode_100.pt checkpoints/checkpoint_episode_500.pt checkpoints/final_checkpoint.pt
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final_checkpoint.pt",
        help="Path to checkpoint file (default: checkpoints/final_checkpoint.pt)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to test (default: 5)"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment visually (opens a window)"
    )
    
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Save gameplay video to file (e.g., output.mp4)"
    )
    
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple checkpoints (provide multiple checkpoint paths)"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        choices=['reinforce', 'ppo'],
        default=None,
        help="Force agent type (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Compare mode
    if args.compare:
        compare_checkpoints(args.compare, num_episodes=args.episodes)
    else:
        # Single checkpoint test
        test_agent(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            render=args.render,
            save_video=args.save_video,
            agent_type=args.type
        )


if __name__ == "__main__":
    main()


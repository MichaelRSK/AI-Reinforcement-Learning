"""
Test and Evaluate Trained Agent

This script loads a trained policy network checkpoint and evaluates
its performance by running it in the Pong environment.

Usage:
    python test_agent.py --checkpoint checkpoints/final_checkpoint.pt
    python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --render
    python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --episodes 10 --render
"""

import argparse
import torch
import numpy as np
import gymnasium as gym
import ale_py
import os
import sys

from train import REINFORCETrainer
from preprocessing import preprocess_frame


def test_agent(
    checkpoint_path,
    num_episodes=5,
    render=False,
    save_video=None,
    max_steps_per_episode=10000
):
    """
    Test a trained agent by running it in the environment.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_episodes: Number of episodes to test
        render: Whether to render the environment visually
        save_video: Optional path to save video (e.g., "output.mp4")
        max_steps_per_episode: Maximum steps per episode (safety limit)
    
    Returns:
        Dictionary with test statistics
    """
    print("=" * 70)
    print("TESTING TRAINED AGENT")
    print("=" * 70)
    
    # Create trainer
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
        sys.exit(1)
    
    print("=" * 70)
    print()
    
    # Set to evaluation mode
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
        stats = test_agent(checkpoint_path, num_episodes=num_episodes, render=False)
        results.append({
            "checkpoint": checkpoint_path,
            "avg_reward": stats["avg_reward"],
            "win_rate": stats["win_rate"]
        })
        print()
    
    # Print comparison
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Checkpoint':<40} {'Avg Reward':<15} {'Win Rate':<10}")
    print("-" * 70)
    for result in results:
        checkpoint_name = os.path.basename(result["checkpoint"])
        print(f"{checkpoint_name:<40} {result['avg_reward']:>10.2f}     {result['win_rate']:>6.1f}%")
    print("=" * 70)


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test a trained Pong agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python test_agent.py --checkpoint checkpoints/final_checkpoint.pt
  
  # Test with visual rendering
  python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --render
  
  # Test for 10 episodes
  python test_agent.py --checkpoint checkpoints/final_checkpoint.pt --episodes 10
  
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
            save_video=args.save_video
        )


if __name__ == "__main__":
    main()


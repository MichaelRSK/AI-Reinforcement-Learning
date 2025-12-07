"""
Model Comparison Tool

Compare REINFORCE vs PPO performance for the Pong environment.
Generates comparison plots and statistics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os


def load_training_stats(stats_file):
    """Load training statistics from JSON file."""
    if not os.path.exists(stats_file):
        print(f"Warning: {stats_file} not found")
        return None
    
    with open(stats_file, 'r') as f:
        return json.load(f)


def smooth_curve(values, weight=0.9):
    """Exponentially smooth a curve for plotting."""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    
    return smoothed


def calculate_win_rate(rewards, window=100):
    """
    Calculate win rate from rewards.
    Win rate = points scored / total points
    Assuming game goes to 21 points.
    """
    win_rates = []
    
    for i in range(len(rewards)):
        start_idx = max(0, i - window + 1)
        window_rewards = rewards[start_idx:i+1]
        
        # Each episode: reward = points_scored - points_lost
        # If game goes to 21: points_lost = 21, so points_scored = reward + 21
        total_points_scored = sum(max(0, r + 21) for r in window_rewards)
        total_points = len(window_rewards) * 21 * 2  # Each game: 21 + 21 points total
        
        win_rate = (total_points_scored / total_points * 100) if total_points > 0 else 0
        win_rates.append(win_rate)
    
    return win_rates


def compare_algorithms():
    """Generate comparison plots and statistics."""
    
    print("=" * 70)
    print("  MODEL COMPARISON: REINFORCE vs PPO")
    print("=" * 70)
    
    # Load statistics
    reinforce_stats = load_training_stats("checkpoints/training_stats.json")
    ppo_stats = load_training_stats("checkpoints_ppo/ppo_training_stats.json")
    
    if reinforce_stats is None:
        print("ERROR: REINFORCE stats not found. Please run REINFORCE training first.")
        return
    
    # Extract data
    reinforce_episodes = reinforce_stats.get("episode", [])
    reinforce_rewards = reinforce_stats.get("reward", [])
    
    if ppo_stats:
        ppo_episodes = ppo_stats.get("episode", [])
        ppo_rewards = ppo_stats.get("reward", [])
    else:
        ppo_episodes = []
        ppo_rewards = []
        print("\nNote: PPO stats not found. Showing REINFORCE only.")
        print("Run PPO training first: python train_ppo.py\n")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('REINFORCE vs PPO Comparison - Pong', fontsize=16, fontweight='bold')
    
    # Plot 1: Raw Rewards
    ax1 = axes[0, 0]
    if reinforce_rewards:
        ax1.plot(reinforce_episodes, reinforce_rewards, alpha=0.3, color='blue', label='REINFORCE (raw)')
        ax1.plot(reinforce_episodes, smooth_curve(reinforce_rewards), color='blue', linewidth=2, label='REINFORCE (smoothed)')
    if ppo_rewards:
        ax1.plot(ppo_episodes, ppo_rewards, alpha=0.3, color='red', label='PPO (raw)')
        ax1.plot(ppo_episodes, smooth_curve(ppo_rewards), color='red', linewidth=2, label='PPO (smoothed)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Win Rate
    ax2 = axes[0, 1]
    if reinforce_rewards:
        reinforce_win_rates = calculate_win_rate(reinforce_rewards)
        ax2.plot(reinforce_episodes, reinforce_win_rates, color='blue', linewidth=2, label='REINFORCE')
    if ppo_rewards:
        ppo_win_rates = calculate_win_rate(ppo_rewards)
        ax2.plot(ppo_episodes, ppo_win_rates, color='red', linewidth=2, label='PPO')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate (100-episode rolling window)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50% (parity)')
    
    # Plot 3: Moving Average (100 episodes)
    ax3 = axes[1, 0]
    if reinforce_rewards:
        reinforce_ma = [np.mean(reinforce_rewards[max(0, i-99):i+1]) for i in range(len(reinforce_rewards))]
        ax3.plot(reinforce_episodes, reinforce_ma, color='blue', linewidth=2, label='REINFORCE')
    if ppo_rewards:
        ppo_ma = [np.mean(ppo_rewards[max(0, i-99):i+1]) for i in range(len(ppo_rewards))]
        ax3.plot(ppo_episodes, ppo_ma, color='red', linewidth=2, label='PPO')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('100-Episode Moving Average')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Comparative Bar Chart (Final Performance)
    ax4 = axes[1, 1]
    algorithms = []
    final_avg_rewards = []
    final_win_rates = []
    
    if reinforce_rewards and len(reinforce_rewards) >= 100:
        algorithms.append('REINFORCE')
        final_avg_rewards.append(np.mean(reinforce_rewards[-100:]))
        final_win_rates.append(reinforce_win_rates[-1] if reinforce_win_rates else 0)
    
    if ppo_rewards and len(ppo_rewards) >= 100:
        algorithms.append('PPO')
        final_avg_rewards.append(np.mean(ppo_rewards[-100:]))
        final_win_rates.append(ppo_win_rates[-1] if ppo_win_rates else 0)
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    ax4.bar(x - width/2, final_avg_rewards, width, label='Avg Reward (last 100)', color=['blue', 'red'][:len(algorithms)])
    ax4.bar(x + width/2, final_win_rates, width, label='Win Rate %', color=['lightblue', 'lightcoral'][:len(algorithms)])
    ax4.set_ylabel('Value')
    ax4.set_title('Final Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_file = "model_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {output_file}")
    plt.show()
    
    # Print statistics
    print("\n" + "=" * 70)
    print("  DETAILED STATISTICS")
    print("=" * 70)
    
    if reinforce_rewards:
        print("\nREINFORCE Results:")
        print(f"  Total Episodes:        {len(reinforce_rewards)}")
        print(f"  Final Avg Reward:      {np.mean(reinforce_rewards[-100:]):.2f} (last 100)")
        print(f"  Best Episode:          {max(reinforce_rewards):.2f}")
        print(f"  Final Win Rate:        {reinforce_win_rates[-1]:.2f}%" if reinforce_win_rates else "N/A")
        print(f"  Improvement:           {reinforce_win_rates[-1] - reinforce_win_rates[0]:.2f}%" if len(reinforce_win_rates) > 0 else "N/A")
    
    if ppo_rewards:
        print("\nPPO Results:")
        print(f"  Total Episodes:        {len(ppo_rewards)}")
        print(f"  Final Avg Reward:      {np.mean(ppo_rewards[-100:]):.2f} (last 100)")
        print(f"  Best Episode:          {max(ppo_rewards):.2f}")
        print(f"  Final Win Rate:        {ppo_win_rates[-1]:.2f}%" if ppo_win_rates else "N/A")
        print(f"  Improvement:           {ppo_win_rates[-1] - ppo_win_rates[0]:.2f}%" if len(ppo_win_rates) > 0 else "N/A")
    
    if reinforce_rewards and ppo_rewards:
        print("\nComparison:")
        reinforce_final = np.mean(reinforce_rewards[-100:])
        ppo_final = np.mean(ppo_rewards[-100:])
        improvement = ((ppo_final - reinforce_final) / abs(reinforce_final)) * 100
        print(f"  PPO vs REINFORCE:      {improvement:+.1f}% better")
        print(f"  Win Rate Difference:   {ppo_win_rates[-1] - reinforce_win_rates[-1]:+.2f}%")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    compare_algorithms()

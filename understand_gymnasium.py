"""
Understanding Gymnasium Environments - Step 2
This script demonstrates how Gymnasium environments work and helps you understand
the key concepts: reset, step, observation_space, and action_space.
"""

import gymnasium as gym
import ale_py
import numpy as np

def explore_gymnasium():
    """Explore and understand Gymnasium environment basics."""
    
    print("=" * 70)
    print("UNDERSTANDING GYMNASIUM ENVIRONMENTS")
    print("=" * 70)
    print()
    
    # Create the Pong environment
    print("1. CREATING THE ENVIRONMENT")
    print("-" * 70)
    env = gym.make("ALE/Pong-v5")
    print(f"   Environment created: {env}")
    print()
    
    # Explore observation space
    print("2. OBSERVATION SPACE")
    print("-" * 70)
    print(f"   Observation Space: {env.observation_space}")
    print(f"   Type: {type(env.observation_space)}")
    print(f"   Shape: {env.observation_space.shape}")
    print(f"   Data type: {env.observation_space.dtype}")
    print(f"   Min value: {env.observation_space.low.min()}")
    print(f"   Max value: {env.observation_space.high.max()}")
    print("   → This tells us what kind of observations the environment provides.")
    print("   → For Pong: RGB images of size 210×160×3 (height × width × channels)")
    print()
    
    # Explore action space
    print("3. ACTION SPACE")
    print("-" * 70)
    print(f"   Action Space: {env.action_space}")
    print(f"   Type: {type(env.action_space)}")
    print(f"   Number of actions: {env.action_space.n}")
    print(f"   Available actions: {list(range(env.action_space.n))}")
    print("   → This tells us what actions the agent can take.")
    print("   → For Pong: 6 discrete actions (NOOP, FIRE, UP, RIGHT, LEFT, DOWN)")
    print()
    
    # Reset the environment
    print("4. RESET() FUNCTION")
    print("-" * 70)
    observation, info = env.reset()
    print(f"   Reset successful!")
    print(f"   Observation shape: {observation.shape}")
    print(f"   Observation dtype: {observation.dtype}")
    print(f"   Observation min/max: {observation.min()}/{observation.max()}")
    print(f"   Info dict: {info}")
    print("   → reset() returns the initial observation and info dictionary")
    print("   → Use this at the start of each episode")
    print()
    
    # Understand the five outputs from step()
    print("5. STEP() FUNCTION - THE FIVE OUTPUTS")
    print("-" * 70)
    print("   env.step(action) returns a tuple of 5 values:")
    print("   1. observation: Next state/observation from the environment")
    print("   2. reward: Reward received for taking the action")
    print("   3. terminated: True if episode ended due to terminal state")
    print("   4. truncated: True if episode ended due to time limit")
    print("   5. info: Dictionary with additional information")
    print()
    
    # Demonstrate step() with random actions
    print("6. INTERACTING WITH THE ENVIRONMENT (Random Actions)")
    print("-" * 70)
    
    # Reset for a fresh episode
    observation, info = env.reset()
    total_reward = 0
    step_count = 0
    
    print("   Running 10 random steps to demonstrate interaction...")
    print()
    
    for i in range(10):
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step in the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Update counters
        total_reward += reward
        step_count += 1
        
        # Display step information
        print(f"   Step {i+1}:")
        print(f"      Action taken: {action}")
        print(f"      Reward received: {reward}")
        print(f"      Episode terminated: {terminated}")
        print(f"      Episode truncated: {truncated}")
        print(f"      Observation shape: {next_obs.shape}")
        
        # Check if episode ended
        if terminated or truncated:
            print(f"      → Episode ended! Total steps: {step_count}, Total reward: {total_reward}")
            print("      → Resetting environment for next episode...")
            observation, info = env.reset()
            total_reward = 0
            step_count = 0
        else:
            observation = next_obs  # Update observation for next step
        
        print()
    
    print(f"   Total reward accumulated: {total_reward}")
    print()
    
    # Explain key concepts
    print("7. KEY CONCEPTS SUMMARY")
    print("-" * 70)
    print("   • Observation Space: Defines what the agent 'sees'")
    print("     - Pong provides RGB images (210×160×3)")
    print()
    print("   • Action Space: Defines what actions the agent can take")
    print("     - Pong has 6 discrete actions")
    print()
    print("   • reset(): Starts a new episode, returns initial observation")
    print("     - Call this at the beginning of each episode")
    print()
    print("   • step(action): Takes an action, returns 5 values:")
    print("     - observation: Next state")
    print("     - reward: Immediate reward (float)")
    print("     - terminated: Episode ended naturally (bool)")
    print("     - truncated: Episode ended due to time limit (bool)")
    print("     - info: Additional debugging info (dict)")
    print()
    print("   • Episode: One complete game/trial")
    print("     - Starts with reset()")
    print("     - Continues with step() until terminated or truncated")
    print("     - Then reset() again for next episode")
    print()
    
    # Close the environment
    print("8. CLEANUP")
    print("-" * 70)
    env.close()
    print("   Environment closed successfully!")
    print()
    
    print("=" * 70)
    print("UNDERSTANDING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  • You now understand how Gymnasium environments work")
    print("  • You've seen how to interact with the environment using random actions")
    print("  • You understand the five outputs from env.step()")
    print("  • Ready to move on to preprocessing and building the policy network!")

if __name__ == "__main__":
    explore_gymnasium()


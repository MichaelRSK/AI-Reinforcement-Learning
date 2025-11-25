"""
Quick script to demonstrate rendering the Pong environment visually.
This shows how to see the game while it's running.
"""

import gymnasium as gym
import ale_py
import time

def render_pong_demo():
    """Demonstrate rendering the Pong environment."""
    
    print("Creating Pong environment with rendering enabled...")
    
    # Create environment - you can specify render_mode
    env = gym.make("ALE/Pong-v5", render_mode="human")
    
    print("Environment created! A window should appear showing the game.")
    print("Taking 100 random actions to demonstrate...")
    print("(Close the window or press Ctrl+C to stop early)")
    print()
    
    # Reset the environment
    observation, info = env.reset()
    
    total_reward = 0
    step_count = 0
    
    try:
        for i in range(100):
            # Take a random action
            action = env.action_space.sample()
            
            # Step the environment (this will update the visual display)
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Check if episode ended
            if terminated or truncated:
                print(f"Episode ended at step {step_count} with total reward: {total_reward}")
                observation, info = env.reset()
                total_reward = 0
                step_count = 0
            
            # Small delay to make it easier to see (optional)
            time.sleep(0.05)  # 50ms delay = ~20 FPS
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
    
    print(f"\nTotal steps taken: {step_count}")
    print("Closing environment...")
    env.close()
    print("Done!")

if __name__ == "__main__":
    render_pong_demo()


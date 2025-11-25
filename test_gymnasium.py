"""
Simple test script to verify Gymnasium and Pong environment are working correctly.
"""

import gymnasium as gym
# Import ale_py to register the ALE namespace with Gymnasium
import ale_py

def test_pong_environment():
    """Test that the Pong environment loads and runs correctly."""
    print("Testing Gymnasium Pong environment...")
    print("-" * 50)
    
    try:
        # Create the Pong environment
        print("1. Creating Pong-v5 environment...")
        env = gym.make("ALE/Pong-v5")
        print("   ✓ Environment created successfully!")
        
        # Check environment properties
        print("\n2. Environment properties:")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        print(f"   Number of actions: {env.action_space.n}")
        
        # Test reset
        print("\n3. Testing reset()...")
        observation, info = env.reset()
        print(f"   ✓ Reset successful!")
        print(f"   Observation shape: {observation.shape}")
        print(f"   Observation dtype: {observation.dtype}")
        
        # Test a few random steps
        print("\n4. Testing step() with random actions...")
        for i in range(5):
            action = env.action_space.sample()  # Random action
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: action={action}, reward={reward}, terminated={terminated}, truncated={truncated}")
        
        print("\n5. Closing environment...")
        env.close()
        print("   ✓ Environment closed successfully!")
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! Gymnasium is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        print("\nPlease check:")
        print("  - Is gymnasium[atari] installed?")
        print("  - Is gymnasium[accept-rom-license] installed?")
        print("  - Are the Atari ROMs properly set up?")
        raise

if __name__ == "__main__":
    test_pong_environment()


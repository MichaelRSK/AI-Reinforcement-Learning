"""
Policy Network for Pong Reinforcement Learning

This module implements a Convolutional Neural Network (CNN) that takes preprocessed
game frames and outputs action probabilities for the Pong environment.

Architecture:
- Input: (batch_size, 1, 80, 80) preprocessed frames
- Conv2D → ReLU
- Conv2D → ReLU
- Flatten
- Fully Connected → 6 action probabilities
- Softmax (applied in forward pass)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    """
    CNN Policy Network for Pong.
    
    Takes preprocessed frames (80×80 grayscale) and outputs probabilities
    for 6 discrete actions in the Pong environment.
    """
    
    def __init__(self, input_channels=1, num_actions=6):
        """
        Initialize the Policy Network.
        
        Args:
            input_channels: Number of input channels (1 for grayscale)
            num_actions: Number of possible actions (6 for Pong)
        """
        super(PolicyNetwork, self).__init__()
        
        # First convolutional layer
        # Input: (batch, 1, 80, 80)
        # Output: (batch, 32, 38, 38) with padding=1 to preserve size better
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=0
        )
        # After conv1: (80-8)/4 + 1 = 19, but with stride 4: (80-8)/4 + 1 = 19
        # Actually: floor((80-8)/4) + 1 = 19
        
        # Second convolutional layer
        # Input: (batch, 32, 19, 19)
        # Output: (batch, 64, 9, 9)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=0
        )
        # After conv2: floor((19-4)/2) + 1 = 8
        
        # Third convolutional layer (optional, for better feature extraction)
        # Input: (batch, 64, 8, 8)
        # Output: (batch, 64, 6, 6)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        # After conv3: 8-3+1 = 6
        
        # Calculate flattened size: 64 * 6 * 6 = 2304
        self.flatten_size = 64 * 6 * 6
        
        # Fully connected layer
        self.fc = nn.Linear(self.flatten_size, 512)
        
        # Output layer (action probabilities)
        self.action_head = nn.Linear(512, num_actions)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 80, 80) or (batch_size, 80, 80)
               If shape is (batch_size, 80, 80), adds channel dimension automatically.
        
        Returns:
            action_probs: Tensor of shape (batch_size, num_actions) with action probabilities
            log_probs: Tensor of shape (batch_size, num_actions) with log probabilities
        """
        # Handle input shape
        if len(x.shape) == 2:  # (H, W) - single frame, no batch, no channel
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel: (1, 1, H, W)
        elif len(x.shape) == 3:  # (batch, H, W) or (H, W, channels)
            # Assume it's (batch, H, W) and add channel dimension
            x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, H, W)
        # If len(x.shape) == 4, it's already (batch, channels, H, W) - good!
        
        # Ensure input is float32
        if x.dtype != torch.float32:
            x = x.float()
        
        # First convolutional layer + ReLU
        x = F.relu(self.conv1(x))
        # Shape: (batch, 32, 19, 19)
        
        # Second convolutional layer + ReLU
        x = F.relu(self.conv2(x))
        # Shape: (batch, 64, 8, 8)
        
        # Third convolutional layer + ReLU
        x = F.relu(self.conv3(x))
        # Shape: (batch, 64, 6, 6)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 2304)
        
        # Fully connected layer + ReLU
        x = F.relu(self.fc(x))
        # Shape: (batch, 512)
        
        # Output layer (logits)
        logits = self.action_head(x)
        # Shape: (batch, num_actions)
        
        # Apply softmax to get probabilities
        action_probs = F.softmax(logits, dim=-1)
        
        # Compute log probabilities (for REINFORCE algorithm)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return action_probs, log_probs
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy given a state.
        
        Args:
            state: Preprocessed frame (numpy array or torch tensor)
                  Shape: (80, 80) or (1, 80, 80) or (batch, 80, 80)
            deterministic: If True, return the most likely action. If False, sample.
        
        Returns:
            action: Integer action index
            log_prob: Log probability of the selected action
        """
        # Convert numpy to torch if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        # Add batch dimension if needed
        if len(state.shape) == 2:  # (80, 80)
            state = state.unsqueeze(0)  # (1, 80, 80)
        elif len(state.shape) == 3 and state.shape[0] != 1:  # (batch, 80, 80)
            pass  # Already has batch dimension
        elif len(state.shape) == 3:  # (1, 80, 80)
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Get probabilities
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            action_probs, log_probs = self.forward(state)
        
        if deterministic:
            # Return most likely action
            action = torch.argmax(action_probs, dim=-1).item()
            log_prob = log_probs[0, action].item()
        else:
            # Sample from the distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            log_prob = log_probs[0, action].item()
        
        return action, log_prob


def test_policy_network():
    """
    Test function to verify the policy network works correctly.
    """
    print("=" * 70)
    print("TESTING POLICY NETWORK")
    print("=" * 70)
    print()
    
    # Create network
    print("1. Creating Policy Network...")
    network = PolicyNetwork(input_channels=1, num_actions=6)
    print(f"   [OK] Network created")
    print(f"   Number of parameters: {sum(p.numel() for p in network.parameters()):,}")
    print()
    
    # Test with dummy input (single frame)
    print("2. Testing forward pass with single frame...")
    dummy_input = torch.randn(1, 1, 80, 80)  # (batch=1, channels=1, height=80, width=80)
    print(f"   Input shape: {dummy_input.shape}")
    
    action_probs, log_probs = network(dummy_input)
    print(f"   Output action_probs shape: {action_probs.shape}")
    print(f"   Output log_probs shape: {log_probs.shape}")
    print(f"   Action probabilities: {action_probs[0].tolist()}")
    print(f"   Probabilities sum: {action_probs[0].sum().item():.6f} (should be ~1.0)")
    print("   [OK] Forward pass successful!")
    print()
    
    # Test with batch input
    print("3. Testing forward pass with batch...")
    batch_input = torch.randn(4, 1, 80, 80)  # Batch of 4 frames
    print(f"   Input shape: {batch_input.shape}")
    
    action_probs, log_probs = network(batch_input)
    print(f"   Output action_probs shape: {action_probs.shape}")
    print(f"   Output log_probs shape: {log_probs.shape}")
    print("   [OK] Batch processing successful!")
    print()
    
    # Test with numpy input (2D, no batch dimension)
    print("4. Testing with numpy array (2D, no batch)...")
    numpy_input = np.random.rand(80, 80).astype(np.float32)
    print(f"   Input shape: {numpy_input.shape}")
    
    action_probs, log_probs = network(torch.from_numpy(numpy_input))
    print(f"   Output action_probs shape: {action_probs.shape}")
    print("   [OK] Numpy input handling successful!")
    print()
    
    # Test get_action method
    print("5. Testing get_action() method...")
    test_state = np.random.rand(80, 80).astype(np.float32)
    
    # Sample action (stochastic)
    action, log_prob = network.get_action(test_state, deterministic=False)
    print(f"   Sampled action: {action}")
    print(f"   Log probability: {log_prob:.4f}")
    
    # Deterministic action
    action_det, log_prob_det = network.get_action(test_state, deterministic=True)
    print(f"   Deterministic action: {action_det}")
    print(f"   Log probability: {log_prob_det:.4f}")
    print("   [OK] get_action() method successful!")
    print()
    
    # Test with preprocessed frame format (from preprocessing.py)
    print("6. Testing with preprocessed frame format...")
    from preprocessing import preprocess_frame
    import gymnasium as gym
    import ale_py
    
    env = gym.make("ALE/Pong-v5")
    observation, info = env.reset()
    
    # Preprocess a real frame
    processed_frame = preprocess_frame(observation, normalize=True)
    print(f"   Preprocessed frame shape: {processed_frame.shape}")
    print(f"   Preprocessed frame dtype: {processed_frame.dtype}")
    print(f"   Preprocessed frame range: [{processed_frame.min():.3f}, {processed_frame.max():.3f}]")
    
    # Get action from preprocessed frame
    action, log_prob = network.get_action(processed_frame, deterministic=False)
    print(f"   Action from real frame: {action}")
    print(f"   Log probability: {log_prob:.4f}")
    print("   [OK] Real frame processing successful!")
    
    env.close()
    print()
    
    print("=" * 70)
    print("[OK] ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("The Policy Network is ready to use in training!")
    print("Next step: Implement the REINFORCE training loop.")


if __name__ == "__main__":
    test_policy_network()


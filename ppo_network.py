"""
PPO Actor-Critic Network for Pong

This module implements the Actor-Critic architecture for PPO:
- Actor: Outputs action probabilities (policy)
- Critic: Outputs state value estimate (baseline)

Both share the same convolutional feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Architecture:
    - Shared convolutional layers for feature extraction
    - Separate heads for actor (policy) and critic (value)
    
    This allows the agent to:
    - Learn a policy (actor)
    - Estimate state values (critic) for variance reduction
    """
    
    def __init__(self, input_channels=1, num_actions=6):
        """
        Initialize the Actor-Critic network.
        
        Args:
            input_channels: Number of input channels (1 for grayscale)
            num_actions: Number of possible actions
        """
        super(PPOActorCritic, self).__init__()
        
        self.num_actions = num_actions
        
        # Shared convolutional feature extractor
        # Input: (batch, 1, 80, 80)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)  # -> (batch, 16, 19, 19)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # -> (batch, 32, 8, 8)
        
        # Calculate flattened size: 32 * 8 * 8 = 2048
        # After conv1: (80 - 8) / 4 + 1 = 19
        # After conv2: (19 - 4) / 2 + 1 = 8
        self.flatten_size = 32 * 8 * 8
        
        # Shared fully connected layer
        self.fc_shared = nn.Linear(self.flatten_size, 256)
        
        # Actor head (policy): outputs action probabilities
        self.actor_fc = nn.Linear(256, num_actions)
        
        # Critic head (value function): outputs single value estimate
        self.critic_fc = nn.Linear(256, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Smaller initialization for policy head (helps with exploration)
        nn.init.orthogonal_(self.actor_fc.weight, gain=0.01)
        # Smaller initialization for value head
        nn.init.orthogonal_(self.critic_fc.weight, gain=1.0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state (batch, 1, 80, 80)
            
        Returns:
            action_probs: Action probabilities (batch, num_actions)
            value: State value estimate (batch, 1)
            log_probs: Log probabilities of actions (batch, num_actions)
        """
        # Ensure input is float and in correct range
        if x.dtype != torch.float32:
            x = x.float()
        
        # Shared feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_shared(x))
        
        # Actor head: compute action probabilities
        action_logits = self.actor_fc(x)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Critic head: compute state value
        value = self.critic_fc(x)
        
        return action_probs, value, log_probs
    
    def get_action(self, state, deterministic=False):
        """
        Select an action given a state.
        
        Args:
            state: Input state (80, 80) or (1, 80, 80) numpy array or tensor
            deterministic: If True, select argmax action; if False, sample
            
        Returns:
            action: Selected action (int)
            log_prob: Log probability of selected action (scalar tensor)
            value: State value estimate (scalar tensor)
        """
        # Convert state to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        # Add channel dimension if needed (80, 80) -> (1, 80, 80)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)  # Add channel dim: (1, 80, 80)
        
        # Add batch dimension if needed (1, 80, 80) -> (1, 1, 80, 80)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)  # Add batch dim: (1, 1, 80, 80)
        
        # Forward pass
        with torch.no_grad():
            action_probs, value, log_probs = self.forward(state)
        
        # Select action
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from the distribution
            action = torch.multinomial(action_probs, num_samples=1).squeeze()
        
        # Get log probability and value for the selected action
        log_prob = log_probs[0, action]
        value = value.squeeze()
        
        return action.item(), log_prob, value
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions for a batch of states (used during training).
        
        Args:
            states: Batch of states (batch, 1, 80, 80) or will add channel dim if (batch, 80, 80)
            actions: Batch of actions (batch,)
            
        Returns:
            action_log_probs: Log probabilities of actions (batch,)
            values: State value estimates (batch,)
            entropy: Entropy of action distributions (batch,)
        """
        # Ensure states have channel dimension (batch, 1, 80, 80)
        if len(states.shape) == 3:
            states = states.unsqueeze(1)  # Add channel dim: (batch, 1, 80, 80)
        
        action_probs, values, log_probs = self.forward(states)
        
        # Get log probabilities for the actual actions taken
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate entropy (for exploration bonus)
        entropy = -(action_probs * log_probs).sum(dim=-1)
        
        # Squeeze value to match batch size
        values = values.squeeze(1)
        
        return action_log_probs, values, entropy


def test_network():
    """Test the PPO Actor-Critic network."""
    print("Testing PPO Actor-Critic Network...")
    
    # Create network
    net = PPOActorCritic(input_channels=1, num_actions=6)
    print(f"Network created with {sum(p.numel() for p in net.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 80, 80)
    
    action_probs, values, log_probs = net(dummy_input)
    print(f"\nForward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Action probs shape: {action_probs.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Action probs sum (should be ~1.0): {action_probs.sum(dim=1)}")
    
    # Test get_action
    single_state = torch.randn(1, 80, 80)
    action, log_prob, value = net.get_action(single_state)
    print(f"\nGet action:")
    print(f"  Action: {action}")
    print(f"  Log prob: {log_prob.item():.4f}")
    print(f"  Value estimate: {value.item():.4f}")
    
    # Test evaluate_actions
    dummy_actions = torch.randint(0, 6, (batch_size,))
    action_log_probs, values, entropy = net.evaluate_actions(dummy_input, dummy_actions)
    print(f"\nEvaluate actions:")
    print(f"  Action log probs shape: {action_log_probs.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Entropy shape: {entropy.shape}")
    print(f"  Mean entropy: {entropy.mean().item():.4f}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_network()

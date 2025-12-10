"""
DQN Model Architecture
Based on "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

Architecture specifications from the paper:
- Input: 84x84x4 (4 stacked grayscale frames)
- Conv1: 32 filters of 8x8, stride 4, ReLU
- Conv2: 64 filters of 4x4, stride 2, ReLU
- Conv3: 64 filters of 3x3, stride 1, ReLU
- FC1: 512 units, ReLU
- Output: num_actions (linear layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network for Atari games.
    
    Args:
        num_actions: Number of possible actions in the environment
        input_channels: Number of input channels (default: 4 for stacked frames)
    """
    
    def __init__(self, num_actions: int, input_channels: int = 4):
        super(DQN, self).__init__()

        self.num_actions = num_actions
        self.input_channels = input_channels

        # Convolutional backbone (paper spec)
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected head (7x7x64 = 3136 flattened features)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, 84, 84)
            
        Returns:
            Q-values for each action, shape (batch_size, num_actions)
        """
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # (B, 64, 7, 7)

        # Flatten conv output (paper spec)
        x = x.view(x.size(0), -1)  # (B, 3136)

        # Fully connected head
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


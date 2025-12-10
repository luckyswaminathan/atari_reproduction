"""
Dueling DQN Architecture
Based on "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)

Architecture:
- Input: 84x84x4 (4 stacked grayscale frames)
- Conv1: 32 filters of 8x8, stride 4, ReLU
- Conv2: 64 filters of 4x4, stride 2, ReLU
- Conv3: 64 filters of 3x3, stride 1, ReLU
- Flatten: 7x7x64 = 3136 features
- Value stream: FC(512) → FC(1) for V(s)
- Advantage stream: FC(512) → FC(num_actions) for A(s,a)
- Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Dueling Deep Q-Network for Atari games.

    Args:
        num_actions: Number of possible actions in the environment
        input_channels: Number of input channels (default: 4 for stacked frames)
    """

    def __init__(self, num_actions: int, input_channels: int = 4):
        super(DQN, self).__init__()

        self.num_actions = num_actions
        self.input_channels = input_channels

        # Convolutional backbone (same as original DQN)
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Dueling architecture: separate value and advantage streams
        # Shared feature layer
        self.fc_hidden = nn.Linear(3136, 512)

        # Value stream: V(s) - single value per state
        self.fc_value = nn.Linear(512, 1)

        # Advantage stream: A(s,a) - advantage per action
        self.fc_advantage = nn.Linear(512, self.num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.

        Args:
            x: Input tensor of shape (batch_size, input_channels, 84, 84)

        Returns:
            Q-values for each action, shape (batch_size, num_actions)
        """
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # (B, 64, 7, 7)

        # Flatten
        x = x.view(x.size(0), -1)  # (B, 3136)

        # Shared hidden layer
        x = F.relu(self.fc_hidden(x))  # (B, 512)

        # Value stream
        value = self.fc_value(x)  # (B, 1)

        # Advantage stream
        advantage = self.fc_advantage(x)  # (B, num_actions)

        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        # This ensures the advantage is zero-meaned
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

"""
Dueling + Multi-Head Attention DQN
Combines dueling architecture with spatial attention.

Architecture:
- Input: 84x84x4 (4 stacked grayscale frames)
- Conv1: 32 filters of 8x8, stride 4, ReLU
- Conv2: 64 filters of 4x4, stride 2, ReLU
- Conv3: 64 filters of 3x3, stride 1, ReLU
- Multi-Head Attention: 4 heads over 7x7=49 spatial tokens (dim=64)
- Attention pooling: mean over tokens → (B, 64)
- Value stream: FC(512) → FC(1) for V(s)
- Advantage stream: FC(512) → FC(num_actions) for A(s,a)
- Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingMHA(nn.Module):
    """
    Dueling DQN with Multi-Head Attention for Atari games.

    Args:
        num_actions: Number of possible actions in the environment
        input_channels: Number of input channels (default: 4 for stacked frames)
    """

    def __init__(self, num_actions: int, input_channels: int = 4):
        super(DuelingMHA, self).__init__()

        self.num_actions = num_actions
        self.input_channels = input_channels

        # Convolutional backbone
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Multi-head attention over spatial tokens (7x7 = 49 tokens, dim=64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=False)

        # Dueling architecture with attention features (64 dim after pooling)
        # Shared feature layer
        self.fc_hidden = nn.Linear(64, 512)

        # Value stream: V(s) - single value per state
        self.fc_value = nn.Linear(512, 1)

        # Advantage stream: A(s,a) - advantage per action
        self.fc_advantage = nn.Linear(512, self.num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling attention network.

        Args:
            x: Input tensor of shape (batch_size, input_channels, 84, 84)

        Returns:
            Q-values for each action, shape (batch_size, num_actions)
        """
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # (B, 64, 7, 7)

        # Attention over spatial tokens
        b, c, h, w = x.shape
        tokens = x.view(b, c, h * w).permute(2, 0, 1)  # (49, B, 64)
        attn_out, _ = self.attn(tokens, tokens, tokens)  # (49, B, 64)
        attn_pooled = attn_out.mean(dim=0)  # (B, 64)

        # Shared hidden layer
        x = F.relu(self.fc_hidden(attn_pooled))  # (B, 512)

        # Value stream
        value = self.fc_value(x)  # (B, 1)

        # Advantage stream
        advantage = self.fc_advantage(x)  # (B, num_actions)

        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

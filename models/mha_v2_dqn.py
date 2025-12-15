"""
Improved MHA DQN - No Information Bottleneck
Fixes the mean pooling issue by keeping full spatial information.

Key improvements over original MHA:
1. No mean pooling - flattens attention output to 3136 features
2. Positional encoding for spatial awareness
3. Residual connection to preserve original features

Architecture:
- Input: 84x84x4 (4 stacked grayscale frames)
- Conv1: 32 filters of 8x8, stride 4, ReLU
- Conv2: 64 filters of 4x4, stride 2, ReLU
- Conv3: 64 filters of 3x3, stride 1, ReLU → (B, 64, 7, 7)
- Positional encoding added to 49 spatial tokens
- Multi-Head Attention: 4 heads over 49 tokens (dim=64)
- Residual connection: attn_out + original tokens
- Flatten: 49 * 64 = 3136 (same as base DQN!)
- FC1: 3136 → 512, ReLU
- Output: 512 → num_actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MHAV2DQN(nn.Module):
    """
    Improved Multi-Head Attention DQN without information bottleneck.
    
    Args:
        num_actions: Number of possible actions in the environment
        input_channels: Number of input channels (default: 4 for stacked frames)
    """

    def __init__(self, num_actions: int, input_channels: int = 4):
        super(MHAV2DQN, self).__init__()

        self.num_actions = num_actions
        self.input_channels = input_channels

        # Convolutional backbone (same as base DQN)
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Positional encoding for 49 spatial positions (7x7 grid)
        self.pos_encoding = nn.Parameter(torch.randn(49, 64) * 0.02)

        # Multi-head attention: 4 heads, embed_dim=64
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=False)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(64)

        # Fully connected head - FULL 3136 features, no bottleneck!
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

        b, c, h, w = x.shape
        
        # Reshape to tokens: (49, B, 64)
        tokens = x.view(b, c, h * w).permute(2, 0, 1)  # (49, B, 64)
        
        # Add positional encoding for spatial awareness
        tokens = tokens + self.pos_encoding.unsqueeze(1)  # (49, B, 64)
        
        # Self-attention over spatial tokens
        attn_out, _ = self.attn(tokens, tokens, tokens)  # (49, B, 64)
        
        # Residual connection + layer norm
        tokens = self.layer_norm(tokens + attn_out)  # (49, B, 64)
        
        # Flatten ALL tokens - NO mean pooling!
        # (49, B, 64) → (B, 49, 64) → (B, 3136)
        x = tokens.permute(1, 0, 2).reshape(b, -1)  # (B, 3136)

        # Fully connected head
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


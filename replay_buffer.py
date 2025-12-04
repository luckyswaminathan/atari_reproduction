"""
Minimal replay buffer for DQN.
- Stores transitions: (state, action, reward, next_state, done)
- Uniform sampling
"""

from collections import deque
import random
from typing import Deque, Tuple

import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(
            maxlen=capacity
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        # Store tensors on CPU to save GPU memory; move during sampling
        self.buffer.append(
            (
                state.detach().cpu(),
                int(action),
                float(reward),
                next_state.detach().cpu(),
                bool(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, device=device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones, device=device, dtype=torch.float32)
        return states, actions, rewards, next_states, dones


"""
DQN Model Variants
"""

from .base_dqn import DQN as BaseDQN
from .dueling_dqn import DQN as DuelingDQN
from .mha_dqn import DQNWithAttention as MHADQN
from .dueling_mha_dqn import DuelingMHA as DuelingMHADQN

__all__ = ["BaseDQN", "DuelingDQN", "MHADQN", "DuelingMHADQN"]

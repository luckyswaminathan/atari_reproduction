"""
DQN Model Variants
"""

from .base_dqn import DQN as BaseDQN
from .dueling_dqn import DQN as DuelingDQN
from .mha_dqn import DQNWithAttention as MHADQN
from .dueling_mha_dqn import DuelingMHA as DuelingMHADQN
from .mha_v2_dqn import MHAV2DQN

__all__ = ["BaseDQN", "DuelingDQN", "MHADQN", "DuelingMHADQN", "MHAV2DQN"]

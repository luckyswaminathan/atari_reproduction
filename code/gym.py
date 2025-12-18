import ale_py  # noqa: F401 - needed to register ALE environments
import gymnasium as gym
from enum import Enum





def main():
    print(gym.pprint_registry())
    env = gym.make("ALE/Breakout-v5")

    class Action(Enum):
        NOOP = 0
        FIRE = 1
        RIGHT = 2
        LEFT = 3

if __name__ == "__main__":
    main()
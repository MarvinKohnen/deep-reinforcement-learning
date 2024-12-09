from enum import Enum
from gymnasium.spaces import Discrete

ActionSpace = Discrete(6)

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5
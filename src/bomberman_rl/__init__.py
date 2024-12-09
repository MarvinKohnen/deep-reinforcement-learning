from gymnasium.envs.registration import register
import pygame

from .envs.gym_wrapper import BombermanEnvWrapper as Bomberman
from .envs.environment import GUI
from .envs.agent_code import *
from .envs.actions import Actions, ActionSpace
from .envs import settings
from .envs import events
from .wrappers import *

pygame.init()
register(
    id="bomberman_rl/bomberman-v0",
    entry_point="bomberman_rl.envs.gym_wrapper:BombermanEnvWrapper",
)
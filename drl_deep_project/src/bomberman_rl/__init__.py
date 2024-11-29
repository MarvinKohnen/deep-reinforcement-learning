from gymnasium.envs.registration import register
import pygame

from .envs.gym_wrapper import BombermanEnvWrapper as Bomberman, Actions
from .envs import settings
from .envs import events

__all__ = ["Bomberman", "Actions", "settings", "events"]

pygame.init()
register(
    id="bomberman_rl/bomberman-v0",
    entry_point="bomberman_rl.envs.gym_wrapper:BombermanEnvWrapper",
)
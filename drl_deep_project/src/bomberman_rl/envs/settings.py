import logging
from pathlib import Path
import pygame

# Game properties
# board size (a smaller board may be useful at the beginning)
COLS = 7
ROWS = 7

SCENARIOS = {
    # official tournament scenario
    "classic": {"TYPE": "BASIC", "CRATE_DENSITY": 0.75, "COIN_COUNT": 9},
    # easier scenarios for Curriculum Learning (i.e. your agent learns easy tasks first)
    "empty": {"TYPE": "BASIC", "CRATE_DENSITY": 0, "COIN_COUNT": 0},
    "coin-heaven": {"TYPE": "BASIC", "CRATE_DENSITY": 0, "COIN_COUNT": 50},
    "loot-crate": {"TYPE": "BASIC", "CRATE_DENSITY": 0.75, "COIN_COUNT": 50},
    "single-coin-fixed": {"TYPE": "SINGLE_COIN", "FIXED": True},
    "single-coin-rand": {"TYPE": "SINGLE_COIN", "FIXED": False},
    "curriculum-coin": {"TYPE": "CURRICULUM", "CRATE": False},
    "curriculum-crate": {"TYPE": "CURRICULUM", "CRATE": True},
    # you might build on the available TYPES
    # you might implement your own custom TYPES in: environment.py -> BombeRLeWorld -> build_arena()
}
MAX_AGENTS = 4

# Round properties
MAX_STEPS = 400


# GUI properties
RENDER_FPS = 8
SCALE = 1
GRID_SIZE = 30 * SCALE
WIDTH = 1000 * SCALE
HEIGHT = 600 * SCALE
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2

ASSET_DIR = Path(__file__).parent.parent / "assets"

AGENT_COLORS = ["green", "blue", "pink", "yellow"]

# Game rules
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2  # = 1 of bomb explosion + N of lingering around

# Rules for agents
TIMEOUT = 0.2
TRAIN_TIMEOUT = float("inf")
REWARD_KILL = 5
REWARD_KILL_SELF = -5
REWARD_COIN = 1

# User input
INPUT_MAP = {
    pygame.K_UP: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_RIGHT: "RIGHT",
    pygame.K_RETURN: "WAIT",
    pygame.K_SPACE: "BOMB",
}

# Logging levels
LOG_GAME = logging.DEBUG
LOG_AGENT_WRAPPER = logging.INFO
LOG_AGENT_CODE = logging.DEBUG
LOG_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

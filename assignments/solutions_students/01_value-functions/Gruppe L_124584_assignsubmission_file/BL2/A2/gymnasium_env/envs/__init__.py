from gymnasium_env.envs.gridworld_env import GridWorldEnv
from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

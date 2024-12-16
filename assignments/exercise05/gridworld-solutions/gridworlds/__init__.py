from gymnasium.envs.registration import register

# Register the example gridworld in gym.
register(
    id='gridworld-v0',
    entry_point='gridworlds.envs:GridWorld'
)

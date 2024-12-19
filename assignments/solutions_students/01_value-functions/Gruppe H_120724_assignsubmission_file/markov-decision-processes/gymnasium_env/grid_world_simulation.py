from datetime import datetime

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

if __name__ == '__main__':
    # register environment
    register(
        id="GymGridWorld-v0",
        entry_point="gymnasium_env.envs.grid_world:GymGridWorld",
    )
    # make env
    env = gym.make('GymGridWorld-v0', runs=25, render_mode='rgb_array')
    # record agent
    env = RecordVideo(
        env,
        video_folder="grid-world-agent-recordings-recordings",
        name_prefix=f"{datetime.now().strftime("%Y%m%d_%H%M%S")}-eval",
        episode_trigger=lambda x: True,
    )
    # record agent stats
    env = RecordEpisodeStatistics(env, buffer_length=1)

    # reset environment
    state = env.reset()

    # simulate a single episode
    episode_over = False
    while not episode_over:
        # frames = env.render()
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Info: {info}")

        episode_over = truncated or done
        if done:
            print("Episode ended.")

    # close environment
    env.close()

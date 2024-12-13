from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

num_eval_episodes: int = 4


if __name__ == "__main__":

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder="cart-pole-agent-recordings-recordings",
        name_prefix=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-policy-2-eval",
        episode_trigger=lambda x: True,
        fps=15
    )
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

    for episode_num in range(num_eval_episodes):

        observation, _ = env.reset()
        cart_pos, cart_velo, pole_angle, pole_angle_velo = observation

        episode_over: bool = False
        while not episode_over:
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            print(f"Action: {action}, State: {state}, Reward: {reward}, Info: {info}")

            episode_over = truncated or done
            if done:
                print("Episode ended.")

    env.close()

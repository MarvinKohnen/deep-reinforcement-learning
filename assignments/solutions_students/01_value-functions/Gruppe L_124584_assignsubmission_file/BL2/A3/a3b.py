from inspect import signature
import gymnasium as gym
from gymnasium import Env
import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from gymnasium import ObservationWrapper
from gymnasium.envs.classic_control import utils
import random

class DiscreteObservation(ObservationWrapper):

    def __init__(self, env: Env, num_bins=4):
        super().__init__(env)
        self._num_bins = num_bins
        self._position_bins = np.linspace(-2.4, 2.4, num=num_bins)
        self._speed_bins = np.linspace(-5, 5, num=num_bins)
        self._angle_bins = np.linspace(-.2095, .2095, num=num_bins)
        self._anglespeed_bins = np.linspace(-5, 5, num=num_bins)
        self._lookup_table = {}
        tmp = np.zeros(shape=(num_bins, num_bins, num_bins, num_bins))
        for idx, v in np.ndenumerate(tmp):
            self._lookup_table[idx] = [[0.9,0.9], [0,0], [0,0], [], 0] #(action_probabilities, count_per_action, cumul_reward, reward_over_time, count)
    
    def observation(self, observation):
        pos = observation[0]
        speed = observation[1]
        angle = observation[2]
        angle_speed = observation[3]

        pos = np.digitize([pos], bins=self._position_bins)[0]
        speed = np.digitize([speed], bins=self._speed_bins)[0]
        angle = np.digitize([angle], bins=self._angle_bins)[0]
        angle_speed = np.digitize([angle_speed], bins=self._anglespeed_bins)[0]
        return [pos, speed, angle, angle_speed]


def main():

    def select_action(probabilities, epsilon, env:Env):
        if np.random.uniform(1.0, 0.0) < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(np.array(list(probabilities)))
    
    def q_function(action_count, reward):
        return reward/action_count

    env = DiscreteObservation(gym.make("CartPole-v1", render_mode="rgb_array"), num_bins=5)
    observation, info = env.reset(seed=42)
    episodes = 500
    epsilon = 0.05

    avg_reward = []

    for i in range(episodes):
        observation, info = env.reset(seed=42)
        episode_over = False
        reward_for_episode = 0
        while not episode_over:
            curr_state = tuple(observation)
            state_data = env._lookup_table[curr_state]
            action = select_action(state_data[0], epsilon, env)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

            state_data[1][action] += 1
            state_data[2][action] += reward
            state_data[0][action] = q_function(action_count=state_data[1][action], reward=state_data[2][action])
            state_data[4] += 1
            env._lookup_table[curr_state] = state_data
            if len(state_data[3]) == 0: state_data[3].append(reward)
            else: state_data[3].append(state_data[3][-1]+reward)
            reward_for_episode += reward



        avg_reward.append(reward_for_episode/500)
    
    
    for agent in list(env._lookup_table.values()):
        x_steps = np.arange(0, agent[4],1)
        plt.plot(x_steps, agent[3], zorder=-agent[4])

    plt.xlabel("Steps")
    plt.ylabel("Cumulative reward")
    plt.title("Cumulative reward over time")
    plt.show()



if __name__ == "__main__":
    main()
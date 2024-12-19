from inspect import signature
import gymnasium as gym
from gymnasium import Env
import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from gymnasium import ObservationWrapper
from gymnasium.envs.classic_control import utils
import random        

def angle_error(last_observation, observation, goal):
    angle = observation[2]
    angle_error = goal - angle
    last_angle = last_observation[2]
    last_angle_error = goal - last_angle
    angle_error_change = angle_error - last_angle_error

    val = 5*angle_error + 100*angle_error_change
    action = 1 if val < 0 else 0
    return action

def angle_anglespeed(observation):
    angle = observation[2]
    angle_speed = observation[3]
    val = angle + angle_speed
    action = 0 if val < 0 else 1
    return action

def angle_speed(observation):
    speed = observation[1]
    angle = observation[2]
    action = 0 if angle < 0 and speed > -1 else 1
    return action

def angle(observation):
    angle = observation[2]
    action = 0 if angle < 0 else 1
    return action

def position(observation):
    pos = observation[0]
    action = 1 if pos < 0 else 0
    return action

    

def main():

    policy_functions = [position, angle, angle_speed, angle_anglespeed, angle_error]

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    avg_rewards = []
    for o, policy in enumerate(policy_functions):
        avg_rewards.append([])
        print(f"Playing with policy {policy.__name__}")
        for i in range(200):
            episode_over = False
            reward_for_episode = 0
            observation, info = env.reset()
            last_observation = deepcopy(observation)
            while not episode_over:
                if len(signature(policy).parameters) == 3:
                    action = policy(last_observation, observation, 0)
                else:
                    action = policy(observation)
                last_observation = deepcopy(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                reward_for_episode += reward
                episode_over = terminated or truncated
            avg_rewards[o].append(reward_for_episode/500)

    env.close()

    print("Generating Plots...")
    for i, policy_selected in enumerate(policy_functions):
        x_points = np.arange(200)
        fig = plt.figure(figsize=(16,9))
        for ii, policy in enumerate(policy_functions):
            if i != ii:
                plt.plot(x_points, avg_rewards[ii], label=policy.__name__, alpha=0.2)
            else:
                plt.plot(x_points, avg_rewards[ii], label=policy.__name__)
        
        plt.xlabel("Episode")
        plt.ylabel("Average reward")
        plt.title("Average rewards over 200 episodes")
        plt.legend()
        #fig.savefig(f"./BL2/A3/Plots/{policy_selected.__name__}.png", dpi=fig.dpi)
        plt.show()
        plt.close()
        
    
    

if __name__ == "__main__":
    main()
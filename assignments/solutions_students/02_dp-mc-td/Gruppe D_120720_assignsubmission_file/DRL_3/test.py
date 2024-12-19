import gridworlds           # import to trigger registration of the environment
import gymnasium as gym
import numpy as np

# create instance
env = gym.make("gridworld-v0")
env.reset()

# test example
sum_rewards = 0
for i in range(10000):
    _, rew, _, _, _ = env.step(env.action_space.sample())
    sum_rewards += rew

print("Summed rewards over 10.000 episodes: ", sum_rewards)
print(f"The grod size is: {env.size}")

def generalized_policy_iteration(env, num_episodes=10000, gamma=0.99, theta=1e-8, max_iter=1000):
    # Initialize random policy
    policy = np.random.choice(env.n_actions, env.n_states)
    
    # Initialize value function
    V = np.zeros(env.n_states)
    
    # Policy evaluation
    def policy_evaluation(V, policy, theta=theta):
        delta = float('inf')
        while delta > theta:
            delta = 0
            for state in range(env.n_states):
                v = 0
                for action in range(env.n_actions):
                    next_states, rewards = env.sample_next_states(state, action)
                    v += np.sum(rewards) + gamma * np.max([V[s] for s in next_states])
                delta = max(delta, abs(v - V[state]))
                V[state] = v
        return V
    
    # Policy improvement
    def policy_improvement(V, policy):
        new_policy = np.argmax(np.array([V[s] + env._get_reward(s, a) for s in range(env.n_states) for a in range(env.n_actions)]).reshape(env.n_states, env.n_actions), axis=1)
        return new_policy
    
    # Main GPI loop
    for _ in range(max_iter):
        V = policy_evaluation(V, policy)
        policy = policy_improvement(V, policy)
        if np.all(policy == policy_improvement(V, policy)):
            break
    
    return policy, V

# Usage
policy, V = generalized_policy_iteration(env)
print(f"Optimal policy: {policy}")
print(f"Value function: {V}")


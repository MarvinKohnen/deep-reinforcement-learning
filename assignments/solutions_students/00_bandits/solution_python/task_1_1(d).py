import numpy as np
import matplotlib.pyplot as plt


# Bandit class is taken from https://github.com/lilianweng/multi-armed-bandit
class BernoulliBandit(object):

    def __init__(self, n, probas=None):
        assert probas is None or len(probas) == n
        self.n = n
        if probas is None:
            self.probas = [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366] # Set values for reproducibility
        else:
            self.probas = probas

        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0


# Parameters
k = 4  # Number of bandit arms
epsilon = 0.1  # Exploration probability
iterations = 1000  # Number of iterations

# Initialization
b = BernoulliBandit(k)
print("Randomly generated Bernoulli bandit has reward probabilities:\n", b.probas)
print("The best machine has index: {} and proba: {}".format(max(range(k), key=lambda i: b.probas[i]), max(b.probas)))


def epsilon_greedy(decay, epsilon): 
    avg_total_reward = np.zeros(iterations)  # Average total reward at each step
    avg_regret = np.zeros(iterations)  # Average regret at each step
    avg_best_arm_ratio = np.zeros(iterations) # Average percentage of best arm choices over 
    og_epsilon = epsilon # Original epsilon

    for i in range(1000):
        # Variables to store results
        Q = np.zeros(k)  # Estimated rewards for each arm (Task 1.1 (a), (b), (d))
        #Q = np.full(k, 1) # (Task 1.1 (c))
        N = np.zeros(k)  # Number of times each arm has been selected
        total_reward = np.zeros(iterations)  # Total reward at each step
        regret = np.zeros(iterations)  # Regret at each step
        best_arm_ratio = np.zeros(iterations) # Percentage of best arm choices over time

        # ε-greedy bandit algorithm
        for t in range(iterations):
            if np.random.random() <= epsilon:
                # Explore: choose a random arm
                action = np.random.randint(k)
            else:
                # Exploit: choose the best arm based on current estimates
                action = np.argmax(Q)

            # Get reward for the chosen action
            reward = b.generate_reward(action)

            # Update the count and the estimated value for the selected arm
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]

            # Calculate best arm ratio
            best_arm_ratio[t] = N[np.argmax(b.probas)] / np.sum(N)
            # print(best_arm_ratio[t])

            # Store total reward and calculate regret
            total_reward[t] = reward if t == 0 else total_reward[t-1] + reward
            regret[t] = (b.best_proba * (t+1)) - total_reward[t]  # Best possible reward - actual reward

            # Decay epsilon based on the amount of past iterations
            if(decay == True):
                epsilon = og_epsilon / (1 + t / 1000) # (Task 1.1 (d))

        avg_total_reward += (total_reward - avg_total_reward) / (i+1)
        avg_regret += (regret - avg_regret) / (i+1)
        avg_best_arm_ratio += (best_arm_ratio - avg_best_arm_ratio) / (i+1)

    return avg_total_reward, avg_regret, avg_best_arm_ratio

# Collect data from greedy epsilon without decay
avg_total_reward, avg_regret, avg_best_arm_ratio = epsilon_greedy(False, epsilon)
# Collect data from greedy epsilon with decay
avg_total_reward_decay, avg_regret_decay, avg_best_arm_ratio_decay = epsilon_greedy(True, epsilon)


# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(avg_total_reward, label="Average Total Reward")
plt.plot(avg_total_reward_decay, label="Average Total Reward with decay", color="green")
plt.xlabel("Iterations")
plt.ylabel("Total Reward")
plt.title("Average Total Reward Over Time")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(avg_regret, label="Average Regret")
plt.plot(avg_regret_decay, label="Average Regret with decay", color="green")
plt.xlabel("Iterations")
plt.ylabel("Regret")
plt.title("Average Total Regret Over Time")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(avg_best_arm_ratio, label="Average Best Arm Ratio")
plt.plot(avg_best_arm_ratio_decay, label="Average Best Arm Ratio with decay", color="green")
plt.xlabel("Iterations")
plt.ylabel("Best Arm Ratio")
plt.title("Average Best Arm Ratio Over Time")
plt.legend()


plt.tight_layout()
plt.show()
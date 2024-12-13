import numpy as np
import matplotlib.pyplot as plt


# Bandit class
class GaussianBandit(object):

    def __init__(self, n, probas=None, mu=0.0, sigma=0.01):
        assert probas is None or len(probas) == n
        self.n = n
        self.mu = mu
        self.sigma = sigma
        if probas is None:
            self.probas = [0.5, 0.5, 0.5, 0.5] # Set initial equal
        else:
            self.probas = probas

        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        # The player selected the i-th machine.ty
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0

    def update_probas(self):
        # Update the probabilities by adding normal distributed random values
        self.probas += np.random.normal(self.mu, self.sigma, self.n)
        self.probas = np.clip(self.probas, 0, 1)  # Ensure probabilities are between 0 and 1
        self.best_proba = max(self.probas)


# Parameters
k = 4  # Number of bandit arms
epsilon = 0.1  # Exploration probability
iterations = 1000  # Number of iterations

# Initialization
b = GaussianBandit(k)
print("Initial expected reward probabilities:\n", b.probas)
print("The best machine has index: {} and proba: {}".format(max(range(k), key=lambda i: b.probas[i]), max(b.probas)))

avg_total_reward = np.zeros(iterations)  # Average total reward at each step
avg_regret = np.zeros(iterations)  # Average regret at each step
avg_best_arm_ratio = np.zeros(iterations) # Average percentage of best arm choices over time
arm_selection_counts = np.zeros((k, iterations))  # Count of actions for each arm


for i in range(1000):
    # Variables to store results
    Q = np.zeros(k)  # Estimated rewards for each arm
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

        # Track arm selection counts
        arm_selection_counts[action, t] += 1

        # Calculate best arm ratio
        best_arm_ratio[t] = N[np.argmax(b.probas)] / np.sum(N)
        # print(best_arm_ratio[t])

        # Store total reward and calculate regret
        total_reward[t] = reward if t == 0 else total_reward[t-1] + reward
        regret[t] = (b.best_proba * (t+1)) - total_reward[t]  # Best possible reward - actual reward

        # Update the probabilities after each step
        b.update_probas()

    avg_total_reward += (total_reward - avg_total_reward) / (i+1)
    avg_regret += (regret - avg_regret) / (i+1)
    avg_best_arm_ratio += (best_arm_ratio - avg_best_arm_ratio) / (i+1)


# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(avg_total_reward, label="Average Total Reward")
plt.xlabel("Iterations")
plt.ylabel("Total Reward")
plt.title("Average Total Reward Over Time")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(avg_regret, label="Average Regret")
plt.xlabel("Iterations")
plt.ylabel("Regret")
plt.title("Average Total Regret Over Time")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(avg_best_arm_ratio, label="Average Best Arm Ratio")
plt.xlabel("Iterations")
plt.ylabel("Best Arm Ratio")
plt.title("Average Best Arm Ratio Over Time")
plt.legend()

plt.tight_layout()
plt.show()

# Arm Selection Distribution Plot
plt.figure(figsize=(10, 6))
for arm in range(k):
    plt.plot(np.cumsum(arm_selection_counts[arm, :]), label=f"Arm {arm + 1}")
plt.xlabel("Iterations")
plt.ylabel("Cumulative Selections")
plt.title("Cumulative Selections of Each Arm Over Time")
plt.legend()
plt.grid()
plt.show()


# Summary of results
print("Estimated action values:", Q)
print("Optimal arm selected:", np.argmax(Q))
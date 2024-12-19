import gridworlds           # import to trigger registration of the environment
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# create instance
env = gym.make("gridworld-v0")
env.reset()

actionValues = np.zeros((5, 5, 4))
epsilon = 0.2
gamma = 0.9
alpha = 0.1
#actionValues.fill(np.random.rand()) # Give each cell a random value in (0,1)

# Target policy
def greedyPolicy(state: np.ndarray):
    x = state[1]
    y = state[0]
    possibleValues = []
    possibleActions = []
    if y > 0:
        possibleValues.append(actionValues[y, x, 0])
        possibleActions.append(0)
    if x < 4:
        possibleValues.append(actionValues[y, x, 1])
        possibleActions.append(1)
    if y < 4:
        possibleValues.append(actionValues[y, x, 2])
        possibleActions.append(2)
    if x > 0:
        possibleValues.append(actionValues[y, x, 3])
        possibleActions.append(3)
    
    action = possibleActions[np.argmax(possibleValues)]
    return action

# Behaviour policy, epsilon-greedy
def epsilonGreedyPolicy(state: np.ndarray):
    if np.random.rand() > epsilon:
        action = env.action_space.sample()
    else:
        action = greedyPolicy(state)
    return action


maxEpisodes = 1000
rewardHistory = []

# Q-learning algorithm
reward = 0
for i in range(maxEpisodes):
    print("Q-learning episode number ", i+1)
    done, truncated = False, False
    obs = env.reset()
    state = obs[0]
    while not done and not truncated:
        action = epsilonGreedyPolicy(state)
        newState, rew, done, truncated, _ = env.step(action)
        reward += rew
        actionValues[state[0], state[1], action] += alpha * (rew + gamma * actionValues[newState[0], newState[1], greedyPolicy(newState)] - actionValues[state[0], state[1], action])
        state = newState
    rewardHistory.append(reward/(i+1))

#print(actionValues)

#############################
# Plotting of the results  #
#############################

cumRewardHistory = np.cumsum(rewardHistory)

height, width = 5,5
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(wspace=0.5)
ax1 = fig.add_subplot(121)
rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(0,5):
    for j in range(0,5):
        if (i+j)%2 == 1:
            rgb_array[i][j] = [255,255,255] 
extent = [0, 5, 0, 5]
im = ax1.imshow(rgb_array, extent=extent, origin='lower', interpolation='None', cmap='viridis')

jump_x = 5 / 10.0
jump_y = 5 / 10.0
x_positions = np.linspace(start=0, stop=5, num=5, endpoint=False)
y_positions = np.linspace(start=0, stop=5, num=5, endpoint=False)

for y_index, y in enumerate(y_positions):
    for x_index, x in enumerate(x_positions):
        for m in range(0,4):
            label = np.round(actionValues[4-y_index, x_index, m], 2)
            text_x = x + jump_x
            text_y = y + jump_y
            if m == 0:
                if (y_index+x_index)%2 == 1:
                    ax1.text(text_x, text_y + 0.25, label, fontsize="small" ,color='black', ha='center', va='center')
                else:
                    ax1.text(text_x, text_y + 0.25, label, fontsize="small", color='white', ha='center', va='center')
            elif m == 1:
                if (y_index+x_index)%2 == 1:
                    ax1.text(text_x + 0.25, text_y, label, fontsize="small", color='black', ha='center', va='center')
                else:
                    ax1.text(text_x + 0.25, text_y, label, fontsize="small", color='white', ha='center', va='center')
            elif m == 2:
                if (y_index+x_index)%2 == 1:
                    ax1.text(text_x, text_y - 0.25, label, fontsize="small", color='black', ha='center', va='center')
                else:
                    ax1.text(text_x, text_y - 0.25, label, fontsize="small", color='white', ha='center', va='center')
            elif m == 3:
                if (y_index+x_index)%2 == 1:
                    ax1.text(text_x - 0.25, text_y, label, fontsize="small", color='black', ha='center', va='center')
                else:
                    ax1.text(text_x - 0.25, text_y, label, fontsize="small", color='white', ha='center', va='center')

ax1.set_title("Action Values")

ax2 = fig.add_subplot(122)
ax2.plot(rewardHistory)
ax2.set_title(f"Epsilon = {epsilon} | Alpha = {alpha} | Gamma: {gamma}")
ax2.set_ylabel("Cumulative Reward per Episode")
ax2.set_xlabel("Episodes")

plt.show()
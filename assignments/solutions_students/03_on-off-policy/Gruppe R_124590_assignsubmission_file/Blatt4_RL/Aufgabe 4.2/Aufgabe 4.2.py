import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CliffWalking-v0', is_slippery=False, render_mode="rgb_array")
env.reset()

actionValuesQ = np.zeros((48, 4))
actionValuesSARSA = np.zeros((48, 4))
epsilon = 0.4 # 0.8, 0.5, 0.2, 0.05, 0 for exercise 4.2b)
gamma = 1
alpha = 0.1

# Target policy Q-learning
def targetPolicy(state):
    action = np.argmax(actionValuesQ[state])
    return action

# Behaviour policy, epsilon-greedy Q-learning
def behaviourPolicy(state: np.ndarray):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = targetPolicy(state)
    return action


# epsilon-greedy policy for SARSA
def SARSAPolicy(state):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(actionValuesSARSA[state])
    return action

# greedy Policy for SARSA
def greedySARSAPolicy(state):
    possibleValues = []
    possibleActions = []
    if state > 11:
        possibleValues.append(actionValuesSARSA[state, 0])
        possibleActions.append(0)
    if not ((state%12) == 11):
        possibleValues.append(actionValuesSARSA[state, 1])
        possibleActions.append(1)
    if state < 36:
        possibleValues.append(actionValuesSARSA[state, 2])
        possibleActions.append(2)
    if not ((state%12) == 0):
        possibleValues.append(actionValuesSARSA[state, 3])
        possibleActions.append(3)
    
    action = possibleActions[np.argmax(possibleValues)]
    return action

returnHistoryQ =[]
returnHistorySARSA =[]
episodes = 500

# Q-learning
rewardQ = 0
for i in range(episodes):
    print("Q-learning episode number ", i+1)
    done, truncated = False, False
    obs = env.reset()
    state = obs[0]
    while not done and not truncated:
        action = behaviourPolicy(state)
        newState, rew, done, truncated, _ = env.step(action)
        rewardQ += rew
        actionValuesQ[state, action] += alpha * (rew + gamma * actionValuesQ[newState, targetPolicy(newState)] - actionValuesQ[state, action])
        state = newState
    returnHistoryQ.append(rewardQ/(i+1))

# SARSA
rewardSARSA = 0
for i in range(episodes):
    print("SARSA episode number ", i+1)
    done, truncated = False, False
    obs = env.reset()
    state = obs[0]
    action = SARSAPolicy(state)
    while not done and not truncated:
        newState, rew, done, truncated, _ = env.step(action)
        rewardSARSA += rew
        nextAction = SARSAPolicy(newState)
        actionValuesSARSA[state, action] += alpha * (rew + gamma * actionValuesSARSA[newState, nextAction] - actionValuesSARSA[state, action])
        state = newState
        action = nextAction
    returnHistorySARSA.append(rewardSARSA/(i+1))


#############################
# Plotting of the results  #
#############################

height, width = 4,12
rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(height):
    for j in range(width):
        if (i+j)%2 == 1:
            rgb_array[i][j] = [255,255,255] 
extent = [0, 12, 0, 4]

jump_x = 5 / 10.0
jump_y = 5 / 10.0
x_positions = np.linspace(start=0, stop=12, num=12, endpoint=False)
y_positions = np.linspace(start=0, stop=4, num=4, endpoint=False)


# plotting Q-learning

fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(wspace=0.5,hspace=1)
ax1 = fig.add_subplot(221)
im = ax1.imshow(rgb_array, extent=extent, origin='lower', interpolation='None', cmap='viridis')

for y_index, y in enumerate(y_positions):
    for x_index, x in enumerate(x_positions):
        dir = targetPolicy(y_index*width+x_index)
        if dir == 0:
            if (y_index+x_index)%2 == 1:
                ax1.arrow(x + 0.5, height-y - 0.5, 0, 0.3, length_includes_head=True, head_width=0.1, head_length=0.1, color='white')
            else:
                ax1.arrow(x + 0.5, height-y - 0.5, 0, 0.3, length_includes_head=True, head_width=0.1, head_length=0.1, color='black')
        elif dir == 1:
            if (y_index+x_index)%2 == 1:
                ax1.arrow(x + 0.5, height-y - 0.5, 0.3, 0, length_includes_head=True, head_width=0.1, head_length=0.1, color='white')
            else:
                ax1.arrow(x + 0.5, height-y - 0.5, 0.3, 0, length_includes_head=True, head_width=0.1, head_length=0.1, color='black')
        elif dir == 2:
            if (y_index+x_index)%2 == 1:
                ax1.arrow(x + 0.5, height-y - 0.5, 0, -0.3, length_includes_head=True, head_width=0.1, head_length=0.1, color='white')
            else:
                ax1.arrow(x + 0.5, height-y - 0.5, 0, -0.3, length_includes_head=True, head_width=0.1, head_length=0.1, color='black')
        elif dir == 3:
            if (y_index+x_index)%2 == 1:
                ax1.arrow(x + 0.5, height-y - 0.5, -0.3, 0, length_includes_head=True, head_width=0.1, head_length=0.1, color='white')
            else:
                ax1.arrow(x + 0.5, height-y - 0.5, -0.3, 0, length_includes_head=True, head_width=0.1, head_length=0.1, color='black')        

ax1.set_title("Optimal Policy Q-learning")

ax2 = fig.add_subplot(222)
ax2.plot(returnHistoryQ)
ax2.set_title(f"Epsilon = {epsilon} | Alpha = {alpha} | Gamma: {gamma}")
ax2.set_ylabel("Cumulative Reward per Episode")
ax2.set_xlabel("Episodes")


# plotting SARSA

ax3 = fig.add_subplot(223)
im = ax3.imshow(rgb_array, extent=extent, origin='lower', interpolation='None', cmap='viridis')


for y_index, y in enumerate(y_positions):
    for x_index, x in enumerate(x_positions):
        dir = greedySARSAPolicy(y_index*width+x_index)
        if dir == 0:
            if (y_index+x_index)%2 == 1:
                ax3.arrow(x + 0.5, height-y - 0.5, 0, 0.3, length_includes_head=True, head_width=0.1, head_length=0.1, color='white')
            else:
                ax3.arrow(x + 0.5, height-y - 0.5, 0, 0.3, length_includes_head=True, head_width=0.1, head_length=0.1, color='black')
        elif dir == 1:
            if (y_index+x_index)%2 == 1:
                ax3.arrow(x + 0.5, height-y - 0.5, 0.3, 0, length_includes_head=True, head_width=0.1, head_length=0.1, color='white')
            else:
                ax3.arrow(x + 0.5, height-y - 0.5, 0.3, 0, length_includes_head=True, head_width=0.1, head_length=0.1, color='black')
        elif dir == 2:
            if (y_index+x_index)%2 == 1:
                ax3.arrow(x + 0.5, height-y - 0.5, 0, -0.3, length_includes_head=True, head_width=0.1, head_length=0.1, color='white')
            else:
                ax3.arrow(x + 0.5, height-y - 0.5, 0, -0.3, length_includes_head=True, head_width=0.1, head_length=0.1, color='black')
        elif dir == 3:
            if (y_index+x_index)%2 == 1:
                ax3.arrow(x + 0.5, height-y - 0.5, -0.3, 0, length_includes_head=True, head_width=0.1, head_length=0.1, color='white')
            else:
                ax3.arrow(x + 0.5, height-y - 0.5, -0.3, 0, length_includes_head=True, head_width=0.1, head_length=0.1, color='black')        

ax3.set_title("Optimal Policy SARSA")

ax4 = fig.add_subplot(224)
ax4.plot(returnHistorySARSA)
ax4.set_title(f"Epsilon = {epsilon} | Alpha = {alpha} | Gamma: {gamma}")
ax4.set_ylabel("Cumulative Reward per Episode")
ax4.set_xlabel("Episodes")

plt.show()

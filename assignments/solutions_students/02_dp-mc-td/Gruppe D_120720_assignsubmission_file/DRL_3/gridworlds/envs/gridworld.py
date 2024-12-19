import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

# Constants for directional movement in the grid
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Define the size of the grid and initialize the reward matrix
n = 5
reward_matrix = np.zeros([n, n])
reward_matrix[0, 1] = 10  # Positive reward at position (0, 1)
reward_matrix[0, 3] = 5  # Positive reward at position (0, 3)


class GridWorld(gym.Env):

    def __init__(
        self,
        reward: np.ndarray = reward_matrix,
        start_state: np.ndarray = np.array([0, 0]),
        upper_steps: np.floating = np.inf,
    ) -> None:
        self.n_states = reward.size  # Total number of states in the grid
        self.n_actions = 4  # Number of possible actions (UP, RIGHT, DOWN, LEFT)
        self.reward_matrix = reward  # Initialize reward matrix
        self.done = False  # Whether the episode is done
        self.start_state = start_state  # Starting position in the grid
        self.upper_steps = upper_steps  # Maximum allowed steps per episode
        self.steps = 0  # Step counter
        self.size = reward.shape[0]  # The size of the grid (n x n)

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, self.size - 1, shape=(2,), dtype=int)

    def step(self, action: np.integer) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action)  # Ensure action is valid
        self.steps += 1

        # End episode if maximum steps reached
        if self.steps >= self.upper_steps:
            self.done = True
            return self.state, self._get_reward(self.state, action), self.done, None

        row, col = self.state  # Current position
        reward = self._get_reward(self.state, action)  # Calculate reward

        # Check for teleportation conditions based on special grid positions
        if row == 0 and col == 1:
            row, col = [4, 1]  # Teleport from (0, 1) to (4, 1)
        elif row == 0 and col == 3:
            row, col = [2, 3]  # Teleport from (0, 3) to (2, 3)
        else:
            # Update position based on action
            if action == UP:
                row = max(row - 1, 0)
            elif action == DOWN:
                row = min(row + 1, self.size - 1)
            elif action == RIGHT:
                col = min(col + 1, self.size - 1)
            elif action == LEFT:
                col = max(col - 1, 0)

        new_state = np.array([row, col])  # New position after action
        self.state = new_state  # Update current state

        return self.state, reward, self.done, False, {}

    def _get_reward(self, state: np.ndarray, action: np.integer) -> float: 
        row, col = state  # Current position
        reward = self.reward_matrix[row, col]  # Base reward from reward matrix

        # Apply penalty if attempting to move outside borders
        if self.at_border() and reward == 0:
            if row == 0 and action == UP:
                reward = -1.0
            if row == self.size - 1 and action == DOWN:
                reward = -1.0
            if col == 0 and action == LEFT:
                reward = -1.0
            if col == self.size - 1 and action == RIGHT:
                reward = -1.0

        return reward

    def at_border(self) -> bool:
        # Check if the agent is at the border of the grid
        row, col = self.state
        return row == 0 or row == self.size - 1 or col == 0 or col == self.size - 1

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray]:
        super().reset(seed=seed)  # Reset environment
        self.steps = 0  # Reset step counter
        self.state = self.start_state  # Reset state to starting state
        self.done = False  # Reset done flag

        return self.state, {}

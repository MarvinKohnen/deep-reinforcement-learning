import gymnasium as gym
import numpy as np
from copy import copy, deepcopy
import pygame
import uuid

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode = None, size: int = 5, custom_transitions: dict = None):
        # The size of the square grid
        self.size = size
        self.window_size = 512

        if custom_transitions == None:
            self._custom_transitions = {}
        else:
            self._custom_transitions = custom_transitions

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)

        self._cumulative_reward = 0

        self._rewards_over_time = []

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # down
            1: np.array([0, 1]),  # right
            2: np.array([-1, 0]),  # up
            3: np.array([0, -1]),  # left
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location}
    
    def _get_info(self):
        return {"cumulative reward": self._cumulative_reward, "rewards over time": self._rewards_over_time}
    
    def reset(self, seed : int = None, options : dict = None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._cumulative_reward = 0
        self._rewards_over_time = []

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            pygame.image.save(self.window, f"./A2_0.jpeg")

        return observation, info
    
    def step(self, action):
        direction = self._action_to_direction[action]
        
        reward = 0

        if (self._agent_location[0], self._agent_location[1]) in list(self._custom_transitions.keys()):
            transition_data = self._custom_transitions[tuple(self._agent_location)]
            new_location = transition_data[0]
            action_reward = transition_data[1]
            self._agent_location = np.array(new_location)
            reward = action_reward
        else:
            old_location = deepcopy(self._agent_location)
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )
            if old_location is self._agent_location:
                reward = -1
        self._rewards_over_time.append(reward)
        self._cumulative_reward += reward

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            pygame.image.save(self.window, f"./A2_{uuid.uuid4()}.jpeg")

        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.flip(self._agent_location) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
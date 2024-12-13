from enum import Enum
from typing import Optional

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pygame


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class GymGridWorld(gym.Env):
    """GridWorld environment for the Gymnasium library"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, runs: int = 1000, render_mode: Optional[str] = 'rgb_array') -> None:
        """
        Constructor
        :param runs: How many runs the agent should perform per
        """
        self.grid_size: int = 5
        self.runs: int = runs

        # define agent's position as well as A/B and A'/B'
        self._agent_location: npt.NDArray[np.int64] = np.array([0, 0], dtype=np.int64)
        self._A_location: npt.NDArray[np.int64] = np.array([1, 0], dtype=np.int64)
        self._A_prime_location: npt.NDArray[np.int64] = np.array([1, 4], dtype=np.int64)
        self._B_location: npt.NDArray[np.int64] = np.array([3, 0], dtype=np.int64)
        self._B_prime_location: npt.NDArray[np.int64] = np.array([3, 2], dtype=np.int64)

        # observations are dictionaries with the agent's and the target's location.
        # each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            # TODO: only incorporate agent's position, remove others
            {
                "agent": gym.spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=np.int64),
                "A": gym.spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=np.int64),
                "B": gym.spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=np.int64),
                "A_prime": gym.spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=np.int64),
                "B_prime": gym.spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=np.int64)
            }
        )

        # we have 4 actions, corresponding to "N", "E", "S", "W"
        self.action_space = gym.spaces.Discrete(4)
        # dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),  # right
            Actions.UP.value: np.array([0, 1]),     # up
            Actions.LEFT.value: np.array([-1, 0]),  # left
            Actions.DOWN.value: np.array([0, -1]),  # down
        }

        # set rewards of the environment
        self.rewards: npt.NDArray[np.int64] = np.zeros((self.grid_size, self.grid_size), dtype=np.int64)
        self.rewards[*self._A_location] = 10     # Field A
        self.rewards[*self._B_location] = 5      # Field B

        self.reward: np.int64 = np.int64(0)

        # init termination variables
        self.done: bool = False
        self.step_counter: int = 0

        # visualization initialization
        pygame.font.init()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.window_size = 512  # The size of the PyGame window
        self.screen = None

    def _get_obs(self) -> dict[str, npt.NDArray[np.int64]]:
        """Auxiliary function to get the agent's location as an observation from the environment"""
        return {
            "agent": self._agent_location,
            "A": self._A_location,
            "A_prime": self._A_prime_location,
            "B": self._B_location,
            "B_prime": self._B_prime_location
        }

    def _get_info(self) -> dict[str, int]:
        """Auxiliary function to get additional information"""
        return {
            "n_runs": self.runs,
            "run_counter": self.step_counter
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        """
        Resets the environment and returns the agent's location.
        :param seed: Seed for PRNG. Optional.
        :param options: Additional options to pass to the environment. Optional
        :return: Agent's location
        """
        super().reset(seed=seed)

        # choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.grid_size, size=2, dtype=np.int64)
        # reset its reward
        self.reward = np.int64(0)
        # reset termination criterion
        self.done = False
        self.step_counter = 0
        # return current agents position
        return self._get_obs(), self._get_info()

    def _get_next_state(self, next_state: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], np.int64]:
        """
        Get the next agent's state by considering grid bounds.
        :param next_state: Potential next state the agent is entering
        :return: agent's next state and reward
        """
        # check if agent tried to step out of the environment
        if not ((0 <= next_state[0] <= self.grid_size - 1) and (0 <= next_state[1] <= self.grid_size - 1)):
            # penalize agent
            reward = np.int64(-1)
            # reset next state to current agent's position
            next_state = self._agent_location
        else:
            # check if agent is currently on A or B, then set next state appropriately
            if np.array_equal(self._agent_location, self._A_location):
                reward = self.rewards[*self._A_location].astype(int)
                next_state = self._A_prime_location
            elif np.array_equal(self._agent_location, self._B_location):
                reward = self.rewards[*self._B_location].astype(int)
                next_state = self._B_prime_location
            else:
                # otherwise, get reward based on the position
                reward = self.rewards[next_state[0], next_state[1]].astype(int)
        # make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            next_state, 0, self.grid_size - 1
        )
        return next_state, reward

    def step(self, action: int) -> tuple[dict[str, npt.NDArray], np.int64, bool, bool, dict[str, int]]:
        """
        Performs an action step in the environment.
        :param action: Encoded action as integer.
        :return: observation, reward, done, truncated, info
        """
        # set truncated variable
        truncated: bool = False
        reward: np.int64 = np.int64(-1)
        try:
            # map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self._action_to_direction[action]
            # compute next state
            next_state, reward = self._get_next_state(self._agent_location + direction)
            # increment step counter
            self.step_counter += 1
            # check if termination condition is met
            if self.step_counter > self.runs:
                self.done = True
            # move agent
            self._agent_location = next_state
            # update the agent's reward
            self.reward = reward
        except KeyError or ValueError:
            truncated = True

        return self._get_obs(), reward, self.done, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        font = pygame.font.Font(None, 24)
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        # First we draw the A and B locations
        rect_A = pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._A_location,
                (pix_square_size, pix_square_size),
            ),
        )

        rect_B = pygame.draw.rect(
            canvas,
            (250, 111, 5),
            pygame.Rect(
                pix_square_size * self._B_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # now visualize A' and B' locations
        rect_A_prime = pygame.draw.rect(
            canvas,
            (250, 95, 95),
            pygame.Rect(
                pix_square_size * self._A_prime_location,
                (pix_square_size, pix_square_size),
            ),
        )

        rect_B_prime = pygame.draw.rect(
            canvas,
            (250, 168, 105),
            pygame.Rect(
                pix_square_size * self._B_prime_location,
                (pix_square_size, pix_square_size),
            ),
        )
        for rect, label in zip([rect_A, rect_B, rect_A_prime, rect_B_prime], ["A", "B", "A'", "B'"]):
            text_surface = font.render(label, True, (0, 0, 0))  # Black text
            text_rect = text_surface.get_rect(center=rect.center)
            canvas.blit(text_surface, text_rect)

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
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


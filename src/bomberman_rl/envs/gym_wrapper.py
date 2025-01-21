import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Space, Discrete, MultiBinary, MultiDiscrete, Sequence, Dict, Text

from . import settings as s
from .environment import GUI, BombeRLeWorld
from .actions import ActionSpace, Actions
from .state_space import observation_space, legacy2gym
from .items import loadScaledAvatar


class BombermanEnvWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": s.RENDER_FPS}

    def __init__(self, args):
        self.args = args
        self.render_mode = args.render_mode
        self.passive = args.passive

        # Delegate
        agents = []
        for player_name in (args.players if args.players else []):
            agents.append((player_name, 0))
        for player_name in (args.learners if args.learners else []):
            agents.append((player_name, 1))
        if not self.passive:
            agents = [("env_user", 0)] + agents
        self.delegate = BombeRLeWorld(self.args, agents)

        # Rendering
        self.window = None
        self.clock = None
        self.gui = None
        assert not (
            self.args.user_play and self.args.no_gui
        ), "User play only possible with GUI enabled"
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        self.user_input = None
        if self.render_mode in ["human", "rgb_array"]:
            self.gui = GUI(self.delegate)  # delegate rendering

        # Gymnasium environment interface
        self.action_space = ActionSpace
        self.observation_space = observation_space()


    def get_user_action(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, True
            elif event.type == pygame.KEYDOWN:
                key_pressed = event.key
                if key_pressed in (pygame.K_q, pygame.K_ESCAPE):
                    return None, True
                elif key_pressed in s.INPUT_MAP:
                    action = s.INPUT_MAP[key_pressed]
                    action = Actions._member_map_[action].value
                    return action, False
        return None, False

    def get_user_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_q,
                pygame.K_ESCAPE,
            ):
                return True
        return False

    def _get_obs(self):
        if not self.passive:
            return legacy2gym(self.delegate.get_state_for_agent(self.delegate.agents[0])) # agent 0 is implicitly the exterior agent
        elif self.delegate.active_agents:
            return legacy2gym(self.delegate.get_state_for_agent(self.delegate.active_agents[0]))
        else:
            return None

    def _get_info(self):
        return {
            "events": self.delegate.agents[0].events, # agent 0 is implicitly the exterior agent
            "leaderboard": self.delegate.leaderboard()
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=self.args.seed)
        self.delegate.new_round()
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        self.delegate.do_step(action)
        terminated = not self.delegate.running
        reward = 0  # TODO reward from state infrastructure
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((s.WIDTH, s.HEIGHT))
            pygame.display.set_caption(f"BombeRLe | Round #{self.delegate.round}")
            icon = loadScaledAvatar(s.ASSET_DIR / "bomb_yellow.png")
            pygame.display.set_icon(icon)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = self.gui.render()

        if self.render_mode == "human":            
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.delegate.running:
            self.delegate.end_round()
        if self.window is not None and not self.args.no_gui:
            pygame.display.quit()
            pygame.quit()

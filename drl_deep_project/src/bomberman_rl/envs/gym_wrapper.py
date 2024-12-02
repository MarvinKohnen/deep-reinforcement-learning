from enum import Enum
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Space, Discrete, MultiBinary, MultiDiscrete, Sequence, Dict, Text

from . import settings as s
from .environment import GUI, BombeRLeWorld


class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5


class BombermanStateWrapper(Space):
    pass  # TODO: interface for state in order to access env.state_space e.g. env.state_space.sample()


class BombermanEnvWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, args):
        self.args = args
        self.render_mode = args.render_mode
        # if args.train == 0 and not args.continue_without_training:
        #     args.continue_without_training = True
        # if args.my_agent:
        #     agents.append((args.my_agent, len(agents) < args.train))
        #     args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
        # for agent_name in args.agents:
        #     agents.append((agent_name, len(agents) < args.train))
        # every_step = not args.skip_frames

        # Delegate
        agents = [("dummy_env_user", 0)] + [
            (opponent, 0) for opponent in args.opponents
        ]
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
        self.action_space = Discrete(6)
        self.observation_space = self._init_observation_space()

    def _multi_discrete_space(self, n=1):
        """
        Arena shaped space
        """
        if n == 1:
            return MultiBinary([s.COLS, s.ROWS])
        else:
            return MultiDiscrete(np.ones((s.COLS, s.ROWS)) * n)
    
    def _init_observation_space(self):
        SInt = Discrete(2 ** 20)
        SWalls = self._multi_discrete_space()
        SCrates = self._multi_discrete_space()
        SCoins = self._multi_discrete_space()
        SBombs = self._multi_discrete_space(s.BOMB_TIMER + 1) # 0 = no bomb
        SExplosions = self._multi_discrete_space(15)
        SAgentPos = self._multi_discrete_space()
        SOpponentsPos = self._multi_discrete_space()
        SAgent = Dict({
            "score": SInt,
            "bombs_left": Discrete(2),
            "position": self._multi_discrete_space()
        })
        SOpponents = Sequence(SAgent)
        return Dict({
            "round": SInt,
            "step": SInt,
            "walls": SWalls,
            "crates": SCrates,
            "coins": SCoins,
            "bombs": SBombs,
            "explosions": SExplosions,
            "self_pos": SAgentPos,
            "opponents_pos": SOpponentsPos,
            "self_info": SAgent,
            "opponents_info": SOpponents
        })


    def _state_delegate2gym(self, state):

        def _agent_delegate2gym(agent, pos):
            return {
                "score": agent[1],
                "bombs_left": int(agent[2]),
                "position": pos
            }
        
        if state is None:
            return None
        
        walls = (state["field"] == - 1).astype("int16")
        crates = (state["field"] == 1).astype("int16")

        coins = np.zeros(state["field"].shape, dtype="int16")
        if len(state["coins"]):
            coins[*zip(*state["coins"])] = 1

        bombs = np.zeros(state["field"].shape, dtype="int16")
        if len(state["bombs"]):
            pos, timer = zip(*state["bombs"])
            pos = list(pos)
            timer_feature = s.BOMB_TIMER - np.array(list(timer))
            bombs[*zip(*pos)] = timer_feature

        self_pos = np.zeros(state["field"].shape, dtype="int16")
        _, _, _, pos = state["self"]
        self_pos[*pos] = 1

        opponents_pos = np.zeros(state["field"].shape, dtype="int16")
        if len(state["others"]):
            positions = [pos for _, _, _, pos in state["others"]]
            opponents_pos[*zip(*positions)] = 1

        self_info = _agent_delegate2gym(state["self"], self_pos)
        
        single_opponents_pos = []
        for _, _, _, pos in state["others"]:
            single_opponent_pos = np.zeros(state["field"].shape, dtype="int16")
            single_opponent_pos[*pos] = 1
            single_opponents_pos.append(single_opponent_pos)
        opponents_info = tuple([_agent_delegate2gym(agent, pos) for agent, pos in zip(state["others"], single_opponents_pos)])

        return {
            "round": state["round"],
            "step": state["step"],
            "walls": walls,
            "crates": crates,
            "coins": coins,
            "bombs": bombs,
            "explosions": state["explosion_map"],
            "self_pos": self_pos,
            "opponents_pos": opponents_pos,
            "self_info": self_info,
            "opponents_info": opponents_info
        }


    def get_user_action(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, True
            elif event.type == pygame.KEYDOWN:
                key_pressed = event.key
                if key_pressed in (pygame.K_q, pygame.K_ESCAPE):
                    return None, True
                elif key_pressed in s.INPUT_MAP:
                    return s.INPUT_MAP[key_pressed], False
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
        # agent 0 is implicitly the exterior agent
        obs = self.delegate.get_state_for_agent(self.delegate.agents[0])
        return self._state_delegate2gym(obs)

    def _get_info(self):
        # agent 0 is implicitly the exterior agent
        return {
            "events": self.delegate.agents[0].events
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

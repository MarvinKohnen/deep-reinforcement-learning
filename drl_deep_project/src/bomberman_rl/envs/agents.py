import importlib
import logging
import multiprocessing as mp
import os
import queue
from collections import defaultdict
from inspect import signature
from time import time
from types import SimpleNamespace
from typing import Any, Tuple
import pygame

from . import events as e
from . import settings as s
from .items import loadScaledAvatar

AGENT_API = {
    "callbacks": {
        "setup": ["self"],
        "act": ["self", "game_state: dict", "**kwargs"],
    },
    "train": {
        "setup_training": ["self"],
        "game_events_occurred": [
            "self",
            "old_game_state: dict",
            "self_action: str",
            "new_game_state: dict",
            "events: List[str]",
        ],
        # "enemy_game_events_occurred": ["self", "enemy_name: str", "old_enemy_game_state: dict", "enemy_action: str", "enemy_game_state: dict", "enemy_events: List[str]"],
        "end_of_round": [
            "self",
            "last_game_state: dict",
            "last_action: str",
            "events: List[str]",
        ],
    },
}

EVENT_STAT_MAP = {
    e.KILLED_OPPONENT: "kills",
    e.KILLED_SELF: "suicides",
    e.COIN_COLLECTED: "coins",
    e.CRATE_DESTROYED: "crates",
    e.BOMB_DROPPED: "bombs",
    e.MOVED_LEFT: "moves",
    e.MOVED_RIGHT: "moves",
    e.MOVED_UP: "moves",
    e.MOVED_DOWN: "moves",
    e.INVALID_ACTION: "invalid",
}


class Agent:
    """
    The Agent game object.

    Architecture:
    In the game process, there is an Agent object that holds the state of the player.
    Via an object of subclassing AgentBackend, it is connected to an AgentRunner instance.

    The Agent calls the callbacks in callbacks.py in the specified code folder by
    calling events on its AgentBackend.
    """

    def __init__(
        self,
        agent_name,
        code_name,
        display_name,
        train: bool,
        backend: "AgentBackend",
        color: str,
        env_user: bool = False,
    ):
        self.backend = backend
        self.name = agent_name
        self.code_name = code_name
        self.display_name = display_name
        self.train = train
        self.env_user = env_user
        self.avatar, self.bomb_sprite, self.shade = self._prepare_avatar(
            color=color
        )
        self.total_score = 0
        self.dead = None
        self.score = None
        self.statistics = None
        self.lifetime_statistics = defaultdict(int)
        self.trophies = None
        self.events = None
        self.available_think_time = None
        self.x = None
        self.y = None
        self.bombs_left = None
        self.last_game_state = None
        self.last_action = None
        self.setup()

    def _prepare_avatar(self, color):
        # Load custom avatar or standard robot avatar of assigned color
        try:
            avatar = loadScaledAvatar(s.ASSET_DIR / ".." / "envs" / "agent_code" / self.code_name / "avatar.png")
        except Exception:
            avatar = loadScaledAvatar(s.ASSET_DIR / f"robot_{color}.png")
        # Load custom bomb sprite
        try:
            bomb = loadScaledAvatar(s.ASSET_DIR / ".." / "envs" / "agent_code" / self.code_name / "bomb.png")
        except Exception:
            bomb = loadScaledAvatar(s.ASSET_DIR / f"bomb_{color}.png")
        # Prepare overlay that will indicate dead agent on the scoreboard
        shade = pygame.Surface((30 * s.SCALE, 30 * s.SCALE), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 208))
        return (avatar, bomb, shade)

    def setup(self):
        # Call setup on backend
        self.backend.send_event("setup")
        self.backend.get("setup")
        if self.train:
            self.backend.send_event("setup_training")
            self.backend.get("setup_training")

    def __str__(self):
        return f"Agent {self.name} under control of {self.code_name}"

    def start_round(self):
        self.dead = False
        self.score = 0
        self.statistics = defaultdict(int)
        self.trophies = []
        self.events = []
        self.available_think_time = self.base_timeout
        self.bombs_left = True
        self.last_game_state = None
        self.last_action = None

    @property
    def base_timeout(self):
        return s.TRAIN_TIMEOUT if self.train else s.TIMEOUT

    def add_event(self, event):
        if event in EVENT_STAT_MAP:
            self.note_stat(EVENT_STAT_MAP[event])
        self.events.append(event)

    def note_stat(self, name, value=1):
        self.statistics[name] += value
        self.lifetime_statistics[name] += value

    def get_state(self):
        """Provide information about this agent for the global game state."""
        return self.name, self.score, self.bombs_left, (self.x, self.y)

    def update_score(self, delta):
        """Add delta to both the current round's score and the total score."""
        self.score += delta
        self.total_score += delta

    def process_game_events(self, game_state):
        self.backend.send_event(
            "game_events_occurred",
            self.last_game_state,
            self.last_action,
            game_state,
            self.events,
        )

    def wait_for_game_event_processing(self):
        self.backend.get("game_events_occurred")

    #    def process_enemy_game_events(self, enemy_game_state, enemy: "Agent"):
    #        self.backend.send_event("enemy_game_events_occurred", enemy.name, enemy.last_game_state, enemy.last_action, enemy_game_state, enemy.events)
    #
    #    def wait_for_enemy_game_event_processing(self):
    #        self.backend.get("enemy_game_events_occurred")

    def store_game_state(self, game_state):
        self.last_game_state = game_state

    def reset_game_events(self):
        self.events = []

    def act(self, game_state, env_user_action=None):
        self.backend.send_event("act", game_state, env_user_action=env_user_action)

    def wait_for_act(self):
        action, think_time = self.backend.get_with_time("act")
        self.note_stat("time", think_time)
        self.note_stat("steps")
        self.last_action = action
        return action, think_time

    def round_ended(self):
        self.backend.send_event(
            "end_of_round", self.last_game_state, self.last_action, self.events
        )
        self.backend.get("end_of_round")

    def render(self, screen, x, y):
        """Draw the agent's avatar to the screen at the given coordinates."""
        screen.blit(self.avatar, (x, y))
        if self.dead:
            screen.blit(self.shade, (x, y))


class AgentRunner:
    """
    Agent callback runner (called by backend).
    """
    def __init__(self, train, agent_name, code_name, result_queue, log_dir):
        self.agent_name = agent_name
        self.code_name = code_name
        self.result_queue = result_queue
        self.agent = importlib.import_module(
            ".envs.agent_code." + self.code_name + ".agent",
            "bomberman_rl"
        ).Agent()
        # self.callbacks = importlib.import_module(
        #     ".envs.agent_code." + self.code_name + ".callbacks",
        #     "bomberman_rl"
        # )
        # if train:
        #     self.train = importlib.import_module(
        #         ".envs.agent_code." + self.code_name + ".train",
        #         "bomberman_rl"
        #     )
        # self.check_interface(train=train)
        # self.fake_self = SimpleNamespace()
        # self.fake_self.train = train
        self.wlogger = logging.getLogger(self.agent_name + "_wrapper")
        self.wlogger.setLevel(s.LOG_AGENT_WRAPPER)
        # self.fake_self.logger = logging.getLogger(self.agent_name + "_code")
        # self.fake_self.logger.setLevel(s.LOG_AGENT_CODE)
        log_dir = f"{log_dir}/{self.code_name}/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        handler = logging.FileHandler(f"{log_dir}{self.agent_name}.log", mode="w")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        self.wlogger.addHandler(handler)
        # self.fake_self.logger.addHandler(handler)

    def check_interface(self, train):
        """
        Check provided agent module's interface for compliance to this env engine
        """
        for module_name in ["callbacks"] + (["train"] if train else []):
            module = getattr(self, module_name)
            for event_name, event_args in AGENT_API[module_name].items():
                proper_signature = f"def {event_name}({', '.join(event_args)}):\n\tpass"
                if not hasattr(module, event_name):
                    raise NotImplementedError(
                        f"Agent code {self.code_name} does not provide callback for {event_name}.\nAdd this function to your code in {module_name}.py:\n\n{proper_signature}"
                    )
                actual_arg_count = len(
                    signature(getattr(module, event_name)).parameters
                )
                event_arg_count = len(event_args)
                if actual_arg_count != event_arg_count:
                    raise TypeError(
                        f"Agent code {self.code_name}'s {event_name!r} has {actual_arg_count} arguments, but {event_arg_count} are required.\nChange your function's signature to the following:\n\n{proper_signature}"
                    )

    def process_event(self, event_name, *event_args, **event_kwargs):
        # module_name = None
        # for module_candidate in AGENT_API:
        #     if event_name in AGENT_API[module_candidate]:
        #         module_name = module_candidate
        #         break
        # if module_name is None:
        #     raise ValueError(f"No information on event {event_name!r} is available")
        # module = getattr(self, module_name)
        try:
            self.wlogger.debug(f"Calling {event_name} on callback.")
            start_time = time()
            event_result = getattr(self.agent, event_name)(*event_args, **event_kwargs)
            duration = time() - start_time
            self.wlogger.debug(
                f"Got result from callback#{event_name} in {duration:.3f}s."
            )
            self.result_queue.put((event_name, duration, event_result))
        except BaseException as e:
            self.wlogger.exception(e)
            self.result_queue.put((event_name, 0, e))


class AgentBackend:
    """
    Base class connecting the agent to a callback implementation.
    """

    def __init__(self, train, agent_name, code_name, result_queue, log_dir):
        self.train = train
        self.code_name = code_name
        self.agent_name = agent_name
        self.result_queue = result_queue
        self.log_dir = log_dir

    def start(self):
        raise NotImplementedError()

    def send_event(self, event_name, *event_args, **event_kwargs):
        raise NotImplementedError()

    def get(self, expect_name: str, block=True, timeout=None):
        return self.get_with_time(expect_name, block, timeout)[0]

    def get_with_time(
        self, expect_name: str, block=True, timeout=None
    ) -> Tuple[Any, float]:
        try:
            event_name, compute_time, result = self.result_queue.get(block, timeout)
            if event_name != expect_name:
                raise ValueError(
                    f"Logic error: Expected result from event {expect_name}, but found {event_name}"
                )
            if isinstance(result, BaseException):
                raise result
            return result, compute_time
        except queue.Empty:
            raise


class SequentialAgentBackend(AgentBackend):
    """
    AgentConnector realised in main thread (easy debugging).
    """

    def __init__(self, train, agent_name, code_name, log_dir):
        super().__init__(train, agent_name, code_name, queue.Queue(), log_dir)
        self.runner = None

    def start(self):
        self.runner = AgentRunner(
            self.train, self.agent_name, self.code_name, self.result_queue, self.log_dir
        )

    def send_event(self, event_name, *event_args, **event_kwargs):
        prev_cwd = os.getcwd()
        os.chdir(
            os.path.dirname(__file__)
            + f'/agent_code/{self.code_name.replace(".", "/")}/'
        )
        try:
            self.runner.process_event(event_name, *event_args, **event_kwargs)
        finally:
            os.chdir(prev_cwd)


QUIT = "quit"


def run_in_agent_runner(
    train: bool,
    agent_name: str,
    code_name: str,
    wta_queue: mp.Queue,
    atw_queue: mp.Queue,
    log_dir: str
):
    runner = AgentRunner(train, agent_name, code_name, atw_queue, log_dir)
    while True:
        event_name, event_args = wta_queue.get()
        if event_name == QUIT:
            break
        runner.process_event(event_name, *event_args)


class ProcessAgentBackend(AgentBackend):
    """
    AgentConnector realised by a separate process (fast and safe mode).
    """

    def __init__(self, train, agent_name, code_name):
        super().__init__(train, agent_name, code_name, mp.Queue())
        self.wta_queue = mp.Queue()
        self.process = mp.Process(
            target=run_in_agent_runner,
            args=(
                self.train,
                self.agent_name,
                self.code_name,
                self.wta_queue,
                self.result_queue,
            ),
        )

    def start(self):
        self.process.start()

    def send_event(self, event_name, *event_args, **event_kwargs):
        self.wta_queue.put((event_name, event_args, event_kwargs))

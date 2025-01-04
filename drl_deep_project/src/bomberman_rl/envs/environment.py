import json
import logging
import pickle
import subprocess
from random import sample
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Tuple
import numpy as np
import pygame

from . import settings as s
from . import events as e
from .actions import Actions
from .state_space import legacy2gym
from .agents import Agent, SequentialAgentBackend
from .items import Bomb, Coin, Explosion, loadScaledAvatar


class Trophy:
    coin_trophy = pygame.transform.smoothscale(
        loadScaledAvatar(s.ASSET_DIR / "coin.png"), (15, 15)
    )
    suicide_trophy = pygame.transform.smoothscale(
        loadScaledAvatar(s.ASSET_DIR / "explosion_0.png"), (15, 15)
    )
    time_trophy = loadScaledAvatar(s.ASSET_DIR / "hourglass.png")


class GenericWorld:
    logger: logging.Logger

    running: bool = False
    step: int
    replay: Dict
    round_statistics: Dict

    agents: List[Agent]
    active_agents: List[Agent]
    arena: np.ndarray
    coins: List[Coin]
    bombs: List[Bomb]
    explosions: List[Explosion]

    round_id: str

    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.colors = list(s.AGENT_COLORS)
        self.round = 0
        self.round_statistics = {}
        self.running = False

    def setup_logging(self):
        self.logger = logging.getLogger("BombeRLeWorld")
        self.logger.setLevel(s.LOG_GAME)
        handler = logging.FileHandler(f"{self.args.log_dir}/game.log", mode="w")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("Initializing game world")

    def new_round(self):
        if self.running:
            self.logger.warning("New round requested while still running")
            self.end_round()

        new_round = self.round + 1
        self.logger.info(f"STARTING ROUND #{new_round}")

        # Bookkeeping
        self.step = 0
        self.bombs = []
        self.explosions = []

        if self.args.match_name is not None:
            match_prefix = f"{self.args.match_name} | "
        else:
            match_prefix = ""
        self.round_id = f'{match_prefix}Round {new_round:02d} ({datetime.now().strftime("%Y-%m-%d %H-%M-%S")})'

        # Arena with wall and crate layout
        self.arena, self.coins, self.active_agents = self.build_arena()
        self.killed_agents = []

        for agent in self.active_agents:
            agent.start_round()

        self.round = new_round
        self.running = True

    def build_arena(self) -> Tuple[np.array, List[Coin], List[Agent]]:
        raise NotImplementedError()

    def add_agent(self, agent_dir, name, train=False, env_user=False):
        assert len(self.agents) < s.MAX_AGENTS

        # if self.args.single_process:
        backend = SequentialAgentBackend(train, name, agent_dir, log_dir=self.args.log_dir)
        # else:
        # backend = ProcessAgentBackend(train, name, agent_dir)
        backend.start()

        color = self.colors.pop()
        agent = Agent(
            name, agent_dir, name, train, backend, color=color, env_user=env_user
        )
        self.agents.append(agent)

    def tile_is_free(self, x, y):
        is_free = self.arena[x, y] == 0
        if is_free:
            for obstacle in self.bombs + self.active_agents:
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free

    def perform_agent_action(self, agent: Agent, action):
        # Perform the specified action if possible, wait otherwise
        try:
            action = Actions(action)._name_
        except ValueError:
            agent.add_event(e.INVALID_ACTION)
        if action == "UP" and self.tile_is_free(agent.x, agent.y + 1):
            agent.y += 1
            agent.add_event(e.MOVED_UP)
        elif action == "DOWN" and self.tile_is_free(agent.x, agent.y - 1):
            agent.y -= 1
            agent.add_event(e.MOVED_DOWN)
        elif action == "LEFT" and self.tile_is_free(agent.x - 1, agent.y):
            agent.x -= 1
            agent.add_event(e.MOVED_LEFT)
        elif action == "RIGHT" and self.tile_is_free(agent.x + 1, agent.y):
            agent.x += 1
            agent.add_event(e.MOVED_RIGHT)
        elif action == "BOMB" and agent.bombs_left:
            self.logger.info(f"Agent <{agent.name}> drops bomb at {(agent.x, agent.y)}")
            self.bombs.append(
                Bomb(
                    (agent.x, agent.y),
                    agent,
                    s.BOMB_TIMER,
                    s.BOMB_POWER,
                    agent.bomb_sprite,
                )
            )
            agent.bombs_left = False
            agent.add_event(e.BOMB_DROPPED)
        elif action == "WAIT":
            agent.add_event(e.WAITED)
        else:
            agent.add_event(e.INVALID_ACTION)

    def poll_and_run_agents(self, env_user_action):
        raise NotImplementedError()

    def send_game_events(self):
        pass

    def do_step(self, env_user_action):
        assert self.running
        self.step += 1
        self.logger.info(f"STARTING STEP {self.step}")
        self.poll_and_run_agents(env_user_action)

        # Progress world elements based
        self.collect_coins()
        self.update_explosions()
        self.update_bombs()
        self.evaluate_explosions()
        self.send_game_events()

        if self.time_to_stop():
            self.end_round()

    def collect_coins(self):
        for coin in self.coins:
            if coin.collectable:
                for a in self.active_agents:
                    if a.x == coin.x and a.y == coin.y:
                        coin.collectable = False
                        self.logger.info(
                            f"Agent <{a.name}> picked up coin at {(a.x, a.y)} and receives 1 point"
                        )
                        a.update_score(s.REWARD_COIN)
                        a.add_event(e.COIN_COLLECTED)
                        a.trophies.append(Trophy.coin_trophy)

    def update_explosions(self):
        # Progress explosions
        remaining_explosions = []
        for explosion in self.explosions:
            explosion.timer -= 1
            if explosion.timer <= 0:
                explosion.next_stage()
                if explosion.stage == 1:
                    explosion.owner.bombs_left = True
            if explosion.stage is not None:
                remaining_explosions.append(explosion)
        self.explosions = remaining_explosions

    def update_bombs(self):
        """
        Count down bombs placed
        Explode bombs at zero timer.

        :return:
        """
        for bomb in self.bombs:
            if bomb.timer > 0:
                bomb.timer -= 1
            else:
                # Explode when timer is finished
                self.logger.info(
                    f"Agent <{bomb.owner.name}>'s bomb at {(bomb.x, bomb.y)} explodes"
                )
                bomb.owner.add_event(e.BOMB_EXPLODED)
                blast_coords = bomb.get_blast_coords(self.arena)

                # Clear crates
                for x, y in blast_coords:
                    self.explosions.append(Explosion(x, y, bomb.owner, s.EXPLOSION_TIMER))
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                        bomb.owner.add_event(e.CRATE_DESTROYED)
                        # Maybe reveal a coin
                        for c in self.coins:
                            if (c.x, c.y) == (x, y):
                                c.collectable = True
                                self.logger.info(f"Coin found at {(x, y)}")
                                bomb.owner.add_event(e.COIN_FOUND)
                bomb.active = False
                # # Create explosion
                # screen_coords = [
                #     (
                #         s.GRID_OFFSET[0] + s.GRID_SIZE * x,
                #         s.GRID_OFFSET[1] + s.GRID_SIZE * y,
                #     )
                #     for (x, y) in blast_coords
                # ]
                # self.explosions.append(
                #     Explosion(
                #         blast_coords, screen_coords, bomb.owner, s.EXPLOSION_TIMER
                #     )
                # )
        self.bombs = [b for b in self.bombs if b.active]

    def evaluate_explosions(self):
        # Explosions
        agents_hit = set()
        for explosion in self.explosions:
            # Kill agents
            if explosion.is_dangerous():
                for a in self.active_agents:
                    if (not a.dead) and (a.x, a.y) == (explosion.x, explosion.y):
                        agents_hit.add(a)
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            self.logger.info(f"Agent <{a.name}> blown up by own bomb")
                            self.logger.info(
                                f"Agent <{a.name}> loses {-s.REWARD_KILL_SELF} points"
                            )
                            a.update_score(s.REWARD_KILL_SELF)
                            a.add_event(e.KILLED_SELF)
                            explosion.owner.trophies.append(Trophy.suicide_trophy)
                        else:
                            self.logger.info(
                                f"Agent <{a.name}> blown up by agent <{explosion.owner.name}>'s bomb"
                            )
                            self.logger.info(
                                f"Agent <{explosion.owner.name}> receives {s.REWARD_KILL} points"
                            )
                            explosion.owner.update_score(s.REWARD_KILL)
                            explosion.owner.add_event(e.KILLED_OPPONENT)
                            explosion.owner.trophies.append(
                                pygame.transform.smoothscale(a.avatar, (15, 15))
                            )

        # Remove hit agents
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
            self.killed_agents.append(a)
            a.add_event(e.GOT_KILLED)
            for aa in self.active_agents:
                aa.add_event(e.OPPONENT_ELIMINATED)

    def end_round(self):
        if not self.running:
            raise ValueError("End-of-round requested while no round was running")
        # Wait in case there is still a game step running
        self.running = False

        for a in self.agents:
            a.note_stat("score", a.score)
            a.note_stat("rounds")
        self.round_statistics[self.round_id] = {
            "steps": self.step,
            **{
                key: sum(a.statistics[key] for a in self.agents)
                for key in ["coins", "kills", "suicides"]
            },
        }

    def time_to_stop(self):
        raise NotImplementedError()

    def end(self):
        if self.running:
            self.end_round()

        # results = {"by_agent": {a.name: a.lifetime_statistics for a in self.agents}}
        # for a in self.agents:
        #     results["by_agent"][a.name]["score"] = a.total_score
        # results["by_round"] = self.round_statistics

        # if self.args.save_stats is not False:
        #     if self.args.save_stats is not True:
        #         file_name = self.args.save_stats
        #     elif self.args.match_name is not None:
        #         file_name = f"results/{self.args.match_name}.json"
        #     else:
        #         file_name = (
        #             f'results/{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.json'
        #         )

        #     name = Path(file_name)
        #     if not name.parent.exists():
        #         name.parent.mkdir(parents=True)
        #     with open(name, "w") as file:
        #         json.dump(results, file, indent=4, sort_keys=True)


class BombeRLeWorld(GenericWorld):
    def __init__(self, args, agents):
        super().__init__(args)
        self.rng = np.random.default_rng(args.seed)
        self.scenario_info = s.SCENARIOS[self.args.scenario]
        self.setup_agents(agents)

    def setup_agents(self, agents):
        # Add specified agents and start their subprocesses
        self.agents = []
        flag_opponent = 0
        for agent_dir, train in agents:
            if list([d for d, t in agents]).count(agent_dir) > 1:
                name = (
                    agent_dir
                    + "_"
                    + str(list([a.code_name for a in self.agents]).count(agent_dir))
                )
            else:
                name = agent_dir.split(".")[-1]
            self.add_agent(agent_dir, name, train=train, env_user=not flag_opponent)
            # Implicitly, first agent is controlled by env user
            flag_opponent += 1

    def build_arena(self):
        if self.scenario_info["TYPE"] != "BASIC":
            return self.build_custom_arena()
        
        WALL = -1
        FREE = 0
        CRATE = 1
        arena = np.zeros((s.COLS, s.ROWS), int)

        # Crates in random locations
        arena[self.rng.random((s.COLS, s.ROWS)) < self.scenario_info["CRATE_DENSITY"]] = (
            CRATE
        )

        # Walls
        arena[:1, :] = WALL
        arena[-1:, :] = WALL
        arena[:, :1] = WALL
        arena[:, -1:] = WALL
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    arena[x, y] = WALL

        # Clean the start positions
        start_positions = [
            (1, 1),
            (1, s.ROWS - 2),
            (s.COLS - 2, 1),
            (s.COLS - 2, s.ROWS - 2),
        ]
        for x, y in start_positions:
            for xx, yy in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if arena[xx, yy] == 1:
                    arena[xx, yy] = FREE

        # Place coins at random, at preference under crates
        coins = []
        all_positions = np.stack(
            np.meshgrid(np.arange(s.COLS), np.arange(s.ROWS), indexing="ij"), -1
        )
        crate_positions = self.rng.permutation(all_positions[arena == CRATE])
        free_positions = self.rng.permutation(all_positions[arena == FREE])
        coin_positions = np.concatenate([crate_positions, free_positions], 0)[
            : self.scenario_info["COIN_COUNT"]
        ]
        for x, y in coin_positions:
            coins.append(Coin((x, y), collectable=arena[x, y] == FREE))

        # Reset agents and distribute starting positions
        active_agents = []
        for agent, start_position in zip(
            self.agents, self.rng.permutation(start_positions)
        ):
            active_agents.append(agent)
            agent.x, agent.y = start_position

        return arena, coins, active_agents
    

    def build_custom_arena(self):
        match self.scenario_info["TYPE"]:
            case "SINGLE_COIN":
                return self.build_single_coin_arena()
            case "CURRICULUM":
                return self.build_curriculum_arena()
            case _:
                raise NotImplementedError(f"Scenario of type {self.scenario_info['TYPE']} not implemented.")
            
    def build_curriculum_arena(self):
        crate = self.scenario_info["CRATE"]

        WALL = -1
        arena = np.zeros((s.COLS, s.ROWS), int)

        # Walls
        arena[:1, :] = WALL
        arena[-1:, :] = WALL
        arena[:, :1] = WALL
        arena[:, -1:] = WALL

        # Start positions
        start_positions = [
            (1, 1),
            (s.COLS - 2, s.ROWS - 2)
        ]
        if s.COLS > 3 and s.ROWS > 3:
            start_positions.extend([
            (s.COLS - 2, 1),
            (1, s.ROWS - 2),
        ])
        assert(len(self.agents)) < len(start_positions), f"This scenario supports only {len(start_positions) - 1} agents for altogether {len(start_positions)} available start positions"
        start_positions = list(self.rng.permutation(start_positions))

        active_agents = []
        for agent, start_position in zip(
            sorted(self.agents, key=lambda a: a.env_user, reverse=True), start_positions
        ):
            active_agents.append(agent)
            agent.x, agent.y = start_position

        env_user = [a for a in self.agents if a.env_user][0]
        x_transpose = env_user.x != 1
        y_transpose = env_user.y != 1
        n_corners = 5
        assert(n_corners < s.ROWS - 2 and n_corners < s.COLS - 2)
        coin_positions = []
        coin_helper_x, coin_helper_y = sorted(sample(range(s.COLS - 2), n_corners)), sorted(sample(range(s.ROWS - 2), n_corners))
        for xi, x in enumerate(coin_helper_x):
            y_min = 0 if xi == 0 else coin_helper_y[xi]
            y_max = s.ROWS - 2 if xi == len(coin_helper_x) - 1 else coin_helper_y[xi + 1]
            for y in range(y_min, y_max):
                coin_positions.append((s.COLS - 1 - (x + 1) if x_transpose else (x + 1), s.ROWS - 1 - (y + 1) if y_transpose else (y + 1)))
            z_max = s.COLS - 2 if xi == len(coin_helper_x) - 1 else coin_helper_x[xi + 1]
            for z in range(x, z_max):
                coin_positions.append((s.COLS - 1 - (z + 1) if x_transpose else (z + 1), s.ROWS - 1 - y_max if y_transpose else y_max))

        coin_positions = list(set(coin_positions))
        coin_positions = [p for p in coin_positions if not p == (env_user.x, env_user.y)]
        coins = [Coin(c, collectable=not crate) for c in coin_positions]
        if crate:
            arena[*zip(*coin_positions)] = 1

        return arena, coins, active_agents
            
    def build_single_coin_arena(self):
        fixed = self.scenario_info["FIXED"]

        WALL = -1
        arena = np.zeros((s.COLS, s.ROWS), int)

        # Walls
        arena[:1, :] = WALL
        arena[-1:, :] = WALL
        arena[:, :1] = WALL
        arena[:, -1:] = WALL

        # Start positions
        start_positions = [
            (1, 1),
            (s.COLS - 2, s.ROWS - 2)
        ]
        if s.COLS > 3 and s.ROWS > 3:
            start_positions.extend([
            (s.COLS - 2, 1),
            (1, s.ROWS - 2),
        ])
        assert(len(self.agents)) < len(start_positions), f"This scenario supports only {len(start_positions) - 1} agents for altogether {len(start_positions)} available start positions"

        if not fixed:
            start_positions = list(self.rng.permutation(start_positions))

        coins = [Coin(start_positions.pop(0), collectable=True)]
        active_agents = []
        for agent, start_position in zip(
            sorted(self.agents, key=lambda a: a.env_user, reverse=True), start_positions
        ):
            active_agents.append(agent)
            agent.x, agent.y = start_position

        return arena, coins, active_agents

    def get_state_for_agent(self, agent: Agent):
        if agent.dead:
            return None

        state = {
            "round": self.round,
            "step": self.step,
            "field": np.array(self.arena),
            "self": agent.get_state(),
            "others": [
                other.get_state() for other in self.agents if other is not agent # in self.active_agents if other is not agent
            ],
            "bombs": [bomb.get_state() for bomb in self.bombs],
            "coins": [coin.get_state() for coin in self.coins if coin.collectable],
        }

        explosion_map = np.zeros(state["field"].shape, dtype="int16")
        for (x, y), stage, timer in [e.get_state() for e in self.explosions]:
            explosion_map[x, y] = (1 - stage) * 10 + timer

        state["explosion_map"] = explosion_map
        return state
    
    def leaderboard(self):
        sorted_agents = [(a.name, a.score) for a in self.agents]
        sorted_agents.sort(reverse=True, key=lambda a: a[1])
        result = {
            name: score for name, score in sorted_agents
        }
        return result

    def poll_and_run_agents(self, env_user_action):
        for a in self.active_agents:
            state = self.get_state_for_agent(a)
            state = legacy2gym(state)
            a.store_game_state(state)
            a.reset_game_events()
            if a.available_think_time > 0:
                a.act(state, env_user_action=env_user_action)

        # Give agents time to decide
        perm = self.rng.permutation(len(self.active_agents))
        for i in perm:
            a = self.active_agents[i]
            if a.available_think_time > 0:
                try:
                    action, think_time = a.wait_for_act()
                except KeyboardInterrupt:
                    # Stop the game
                    raise
                except:
                    if not self.args.silence_errors:
                        msg = f"Exception raised by Agent <{a.name}>. Agent set to passive for the rest of the episode."
                        self.logger.error(msg)
                        print(msg)
                        raise
                    # Agents with errors cannot continue
                    action = "ERROR"
                    think_time = float("inf")

                self.logger.info(
                    f"Agent <{a.name}> chose action {action} in {think_time:.2f}s."
                )
                if think_time > a.available_think_time:
                    next_think_time = a.base_timeout - (
                        think_time - a.available_think_time
                    )
                    self.logger.warning(
                        f'Agent <{a.name}> exceeded think time by {think_time - a.available_think_time:.2f}s. Setting action to "WAIT" and decreasing available time for next round to {next_think_time:.2f}s.'
                    )
                    action = Actions._member_map_["WAIT"]
                    a.trophies.append(Trophy.time_trophy)
                    a.available_think_time = next_think_time
                else:
                    self.logger.info(
                        f"Agent <{a.name}> stayed within acceptable think time."
                    )
                    a.available_think_time = a.base_timeout
            else:
                self.logger.info(
                    f"Skipping agent <{a.name}> because of last slow think time."
                )
                a.available_think_time += a.base_timeout
                action = Actions._member_map_["WAIT"]
            self.perform_agent_action(a, action)

    def send_game_events(self):
        # Send events to all agents that expect them, then reset and wait for them
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.process_game_events(self.get_state_for_agent(a))
                for enemy in self.active_agents:
                    if enemy is not a:
                        pass
                        # a.process_enemy_game_events(self.get_state_for_agent(enemy), enemy)
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.wait_for_game_event_processing()
                for enemy in self.active_agents:
                    if enemy is not a:
                        pass
                        # a.wait_for_enemy_game_event_processing()

    def time_to_stop(self):
        # Check round stopping criteria
        if any(a.env_user for a in self.killed_agents):
            self.logger.info("Env user dead, wrap up round")
            return True

        if not len(self.active_agents):
            self.logger.info("No agent left")
            return True
        
        if (
            len(self.active_agents) == 1
            and (self.arena == 1).sum() == 0
            and all([not c.collectable for c in self.coins])
            and len(self.bombs) + len(self.explosions) == 0
        ):
            self.logger.info("One agent left with nothing to do")
            return True
        
        if (
            self.scenario_info["TYPE"] == "SINGLE_COIN"
            and (self.arena == 1).sum() == 0
            and all([not c.collectable for c in self.coins])):
            return True

        if self.step >= s.MAX_STEPS:
            self.logger.info("Maximum number of steps reached, wrap up round")
            return True

        return False
    
    def end_round(self):
        super().end_round()

        self.logger.info(f"WRAPPING UP ROUND #{self.round}")
        # Clean up survivors
        for a in self.active_agents:
            a.add_event(e.SURVIVED_ROUND)

        # Send final event to agents that expect them
        for a in self.agents:
            if a.train:
                a.round_ended()

    def end(self):
        super().end()
        self.logger.info("SHUT DOWN")
        for a in self.agents:
            # Send exit message to shut down agent
            self.logger.debug(f"Sending exit message to agent <{a.name}>")
            # todo multiprocessing shutdown


class GUI:
    def __init__(self, world: GenericWorld):
        self.world = world
        self.screen = None

        # Font for scores and such
        font_name = s.ASSET_DIR / "emulogic.ttf"
        self.fonts = {
            "huge": pygame.font.Font(font_name, 20 * s.SCALE),
            "big": pygame.font.Font(font_name, 16 * s.SCALE),
            "medium": pygame.font.Font(font_name, 10 * s.SCALE),
            "small": pygame.font.Font(font_name, 8 * s.SCALE),
        }
        self.frame = 0

    def initScreen(self):
        self.screen = pygame.Surface((s.WIDTH, s.HEIGHT))
        self.t_wall = loadScaledAvatar(s.ASSET_DIR / "brick.png")
        self.t_crate = loadScaledAvatar(s.ASSET_DIR / "crate.png")

    def render_text(
        self, text, x, y, color, halign="left", valign="top", size="medium", aa=False
    ):
        text_surface = self.fonts[size].render(text, aa, color)
        text_rect = text_surface.get_rect()
        if halign == "left":
            text_rect.left = x
        if halign == "center":
            text_rect.centerx = x
        if halign == "right":
            text_rect.right = x
        if valign == "top":
            text_rect.top = y
        if valign == "center":
            text_rect.centery = y
        if valign == "bottom":
            text_rect.bottom = y
        self.screen.blit(text_surface, text_rect)

    def render(self):
        if self.screen is None:
            self.initScreen()

        if self.world.round == 0:
            return

        self.screen.fill((0, 0, 0))
        self.frame += 1

        # World
        for x in range(self.world.arena.shape[0]):
            for y in range(self.world.arena.shape[1]):
                if self.world.arena[x, y] == -1:
                    self.screen.blit(
                        self.t_wall,
                        (
                            s.GRID_OFFSET[0] + s.GRID_SIZE * x,
                            s.GRID_OFFSET[1] + s.GRID_SIZE * (s.ROWS - y - 1),
                        ),
                    )
                if self.world.arena[x, y] == 1:
                    self.screen.blit(
                        self.t_crate,
                        (
                            s.GRID_OFFSET[0] + s.GRID_SIZE * x,
                            s.GRID_OFFSET[1] + s.GRID_SIZE * (s.ROWS - y - 1),
                        ),
                    )
        self.render_text(
            f"Step {self.world.step:d}",
            s.GRID_OFFSET[0],
            s.HEIGHT - s.GRID_OFFSET[1] / 2,
            (64, 64, 64),
            valign="center",
            halign="left",
            size="medium",
        )

        # Items
        for bomb in self.world.bombs:
            bomb.render(
                self.screen,
                s.GRID_OFFSET[0] + s.GRID_SIZE * bomb.x,
                s.GRID_OFFSET[1] + s.GRID_SIZE * (s.ROWS - bomb.y - 1),
            )
        for coin in self.world.coins:
            if coin.collectable:
                coin.render(
                    self.screen,
                    s.GRID_OFFSET[0] + s.GRID_SIZE * coin.x,
                    s.GRID_OFFSET[1] + s.GRID_SIZE * (s.ROWS - coin.y - 1),
                )

        # Agents
        for agent in self.world.active_agents:
            agent.render(
                self.screen,
                s.GRID_OFFSET[0] + s.GRID_SIZE * agent.x,
                s.GRID_OFFSET[1] + s.GRID_SIZE * (s.ROWS - agent.y - 1),
            )

        # Explosions
        for explosion in self.world.explosions:
            explosion.render(
                self.screen,
                s.GRID_OFFSET[0] + s.GRID_SIZE * explosion.x,
                s.GRID_OFFSET[1] + s.GRID_SIZE * (s.ROWS - explosion.y - 1),
            )

        # Scores
        # agents = sorted(self.agents, key=lambda a: (a.score, -a.mean_time), reverse=True)
        agents = self.world.agents
        leading = max(agents, key=lambda a: (a.score, a.name))
        y_base = s.GRID_OFFSET[1] + 15 * s.SCALE
        for i, a in enumerate(agents):
            bounce = (
                0
                if (a is not leading or self.world.running)
                else np.abs(10 * s.SCALE * np.sin(5 * time()))
            )
            a.render(
                self.screen,
                600 * s.SCALE,
                y_base + 50 * s.SCALE * i - 15 * s.SCALE - bounce,
            )
            self.render_text(
                a.display_name,
                650 * s.SCALE,
                y_base + 50 * s.SCALE * i,
                (64, 64, 64) if a.dead else (255, 255, 255),
                valign="center",
                size="small",
            )
            for j, trophy in enumerate(a.trophies):
                self.screen.blit(
                    trophy,
                    (
                        660 * s.SCALE + 10 * s.SCALE * j,
                        y_base + 50 * s.SCALE * i + 12 * s.SCALE,
                    ),
                )
            self.render_text(
                f"{a.score:d}",
                830 * s.SCALE,
                y_base + 50 * s.SCALE * i,
                (255, 255, 255),
                valign="center",
                halign="right",
                size="big",
            )
            self.render_text(
                f"{a.total_score:d}",
                890 * s.SCALE,
                y_base + 50 * s.SCALE * i,
                (64, 64, 64),
                valign="center",
                halign="right",
                size="big",
            )

        # End of round info
        if not self.world.running:
            x_center = (
                (s.WIDTH - s.GRID_OFFSET[0] - s.COLS * s.GRID_SIZE) / 2
                + s.GRID_OFFSET[0]
                + s.COLS * s.GRID_SIZE
            )
            color = np.int_(
                (
                    255 * (np.sin(3 * time()) / 3 + 0.66),
                    255 * (np.sin(4 * time() + np.pi / 3) / 3 + 0.66),
                    255 * (np.sin(5 * time() - np.pi / 3) / 3 + 0.66),
                )
            )
            self.render_text(
                leading.display_name,
                x_center,
                320 * s.SCALE,
                color,
                valign="top",
                halign="center",
                size="huge",
            )
            self.render_text(
                "has won the round!",
                x_center,
                350 * s.SCALE,
                color,
                valign="top",
                halign="center",
                size="big",
            )
            leading_total = max(
                self.world.agents, key=lambda a: (a.total_score, a.display_name)
            )
            if leading_total is leading:
                self.render_text(
                    f"{leading_total.display_name} is also in the lead.",
                    x_center,
                    390 * s.SCALE,
                    (128, 128, 128),
                    valign="top",
                    halign="center",
                    size="medium",
                )
            else:
                self.render_text(
                    f"But {leading_total.display_name} is in the lead.",
                    x_center,
                    390 * s.SCALE,
                    (128, 128, 128),
                    valign="top",
                    halign="center",
                    size="medium",
                )
        return self.screen
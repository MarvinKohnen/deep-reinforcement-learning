import math
from random import randint
from multiprocessing import Queue, Process
import pygame
import numpy as np
import logging
import json
import copy
import heapq
from pathlib import Path
from sympy import primerange
from itertools import combinations
from collections import defaultdict
import gymnasium
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecEnv,
    SubprocVecEnv,
)
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from bomberman_rl import settings as s, Actions, Bomberman
from bomberman_rl.wrappers import *

from argparsing import parse
from render import QualificationGUI

N_ENVS = 25
PAIRING_CARDINALITY = 3
LOG_PATH = Path(__file__).parent / "logs" / "tournament"
logger = logging.getLogger(__name__)

def makeEnv(
    args,
    vec_env_cls=SubprocVecEnv,
    name_prefix=""
):
    train_render = {"no_gui": True, "render_mode": None}
    train_args = copy.copy(args)
    train_args.__dict__.update(train_render)
    env_train = make_vec_env(
        lambda: makeSingleEnv(
            train_args,
            name_prefix=name_prefix
        ),
        n_envs=N_ENVS,
        vec_env_cls=vec_env_cls,
    )
    return env_train


def makeSingleEnv(
    args,
    name_prefix="",
):
    pygame.init()
    env = gymnasium.make(
        "bomberman_rl/bomberman-v0", args=args
    )
    env = FixedLengthOpponentsInfo(env, n_opponents=PAIRING_CARDINALITY) # fix sequence length in order for VecEnv to successfully operate on state space
    if args.video:
        env = RecordVideo(env, video_folder=args.video, name_prefix=name_prefix, episode_trigger=lambda _: True)
    return env


def demo(model, env, n_steps=100, deterministic=True):
    if isinstance(env, VecEnv):
        obs = env.reset()
    else:
        obs, _ = env.reset()
    terminated, truncated = False, False
    reward = 0
    for i in range(n_steps):
        if not (terminated or truncated):
            if isinstance(env, VecEnv):
                action = model.predict(obs, deterministic=deterministic)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
                action = action.squeeze()
            obs, r, terminated, truncated, _ = env.step(action)
            reward += r
    #print(f"Demo reward: {reward}")
    env.close()

class DummyModel():
    def __init__(self, env):
        if isinstance(env, VecEnv):
            self.a = np.array([env.action_space.sample() for i in range(env.num_envs)])
        self.a = env.action_space.sample()

    def predict(self, *args, **kwargs):
        return self.a, None

# def qualificationDemo(args, competitors):
#     pygame.init()
#     demo_render = {
#         "players": competitors,
#         "no_gui": False,
#         "video": None,
#         "render_mode": "human",
#     }
#     demo_args = copy.copy(args)
#     demo_args.__dict__.update(demo_render)
#     env_demo = makeSingleEnv(demo_args, name_prefix="qualificationDemo")
#     demo(DummyModel(env_demo), env_demo, n_steps=400)

    
def collectEpisodeResults(dones, infos):
    """Return leaderboards from environments with just finished episode"""
    return [info["leaderboard"] for done, info in zip(dones, infos) if done]


def aggregateScoreboards(scoreboards):
    """Average scores per entry over entry appearances in multiple scoreboards"""
    avg_scores, avg_counters = defaultdict(float), defaultdict(float)
    for board in scoreboards:
        for competitor, score in board.items():
            avg_counters[competitor] = avg_counters[competitor] + 1
            avg_scores[competitor] = avg_scores[competitor] + 1 / avg_counters[
                competitor
            ] * (score - avg_scores[competitor])
        yield dict(avg_scores)


def episodeResult2scoreboard(match_result):
    """Transform raw match scores to ranking based scores"""
    scores = {
        0: 3,  # 3 points for 1st
        1: 1,  # 1 point for 2nd
    }
    # Randomly break ties 
    match_result = {
        k: v + randint(0, 10) * .01 for k, v in match_result.items()
    }
    return {
        competitor_result[0]: scores.get(i, 0)
        for i, competitor_result in enumerate(
            sorted(match_result.items(), key=lambda x: x[1], reverse=True)
        )
    }


def recordMatchingDemo(args, scoreboard):
    """Records episodes until it finds one which result matches the aggregated result"""
    demo_render = {
        "no_gui": False,
        "video": f"{LOG_PATH}/replays/matchingDemo",
        "render_mode": "rgb_array",
    }
    demo_args = copy.copy(args)
    demo_args.__dict__.update(demo_render)
    # env_demo = make_vec_env(lambda: makeSingleEnv(demo_args, [], {}, [], {}), n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env_demo = makeSingleEnv(demo_args, name_prefix="test")
    env_demo.reset()
    finished = False
    while not finished:
        # _, _, dones, infos = env_demo.step([None] * n_envs)
        _, _, terminated, truncated, info = env_demo.step(None)
        # new_episode_results = collectEpisodeResults(dones, infos)
        new_episode_results = [info["leaderboard"]] if terminated or truncated else []
        for episode_result in new_episode_results:
            # Randomly break ties
            episode_result = {
                k: v + randint(0, 10) * .01 for k, v in episode_result.items()
            }
            comparison = list(
                zip(
                    sorted(
                        episode_result.items(),
                        key=lambda x: x[1],
                    ),
                    sorted(scoreboard.items(), key=lambda x: x[1]),
                )
            )
            if all([c[0][0] == c[1][0] for c in comparison]):
                finished = True
            else:
                env_demo.reset()
    env_demo.close()


def match(args, competitors, n_episodes=50, demo=False):
    """Compete agents by running multiple episodes"""
    args.learners = []
    args.players = competitors
    env_train = makeEnv(args=args)

    # Aggregated match results TODO consider seeds?
    env_train.reset()
    episode_results = []
    while len(episode_results) < n_episodes:
        _, _, dones, infos = env_train.step([None] * N_ENVS)
        episode_results.extend(collectEpisodeResults(dones, infos))
    env_train.close()
    episode_scoreboards = (episodeResult2scoreboard(r) for r in episode_results)
    aggregated_episode_scoreboard = list(aggregateScoreboards(episode_scoreboards))[-1]
    if demo:
        recordMatchingDemo(args, scoreboard=aggregated_episode_scoreboard)
    return aggregated_episode_scoreboard


def generatePairings(
    competitors: list[str], n_competitor_pairings
):
    def pairing_hash(pairing):
        return hash(tuple([c for _, c in sorted(pairing, key=lambda x: x[1])]))
    
    n_possible_competitor_pairings = math.comb(
        len(competitors) - 1, PAIRING_CARDINALITY - 1
    )
    exhaustive = n_possible_competitor_pairings <= n_competitor_pairings
    n_competitor_pairings = min(
        n_possible_competitor_pairings, n_competitor_pairings
    )
    if exhaustive:
        for pairing in combinations(competitors, PAIRING_CARDINALITY):
            yield pairing
    else:
        pairing_hashes = set()
        c_counts = []
        for c in competitors:
            heapq.heappush(c_counts, (0, c))
        while c_counts[0][0] < n_competitor_pairings:
            pairing, dropped, done = [], [], False
            for i in range(PAIRING_CARDINALITY):
                pairing.append(heapq.heappop(c_counts))
            while pairing_hash(pairing) in pairing_hashes:
                idx = randint(0, PAIRING_CARDINALITY - 1)
                dropped.append(pairing[idx])
                del pairing[idx]
                try:
                    pairing.append(heapq.heappop(c_counts))
                except IndexError:
                    print("index error")
                    for p, c in dropped:
                        heapq.heappush(c_counts, (p, c))
                    for p, c in pairing:
                        heapq.heappush(c_counts, (p, c))
                    for pairing in combinations(competitors, PAIRING_CARDINALITY):
                        if not pairing_hash(zip(pairing, pairing)) in pairing_hashes:
                            c_counts = [
                                (p + 1, c) if c in pairing else (p, c)
                                for p, c in c_counts
                            ]
                            heapq.heapify(c_counts)
                            pairing_hashes.add(pairing_hash(zip(pairing, pairing)))
                            yield pairing
                            done = True
                            break
                    break
            if not done:
                pairing_hashes.add(pairing_hash(pairing))
                for p, c in dropped:
                    heapq.heappush(c_counts, (p, c))
                for p, c in pairing:
                    heapq.heappush(c_counts, (p + 1, c))
                yield [c for _, c in pairing]


# def test(pairings):
#     #print(sorted(pairings, key=lambda p: math.prod(p)))
#     duplicates = []
#     result = defaultdict(int)
#     for p in pairings:
#         p_hash = hash("".join(sorted(p)))
#         if p_hash in duplicates:
#             raise AssertionError(p)
#         else:
#             duplicates.append(p_hash)
#             for k in p:
#                 result[k] = result[k] + 1
#     return result
# l = list(pairings(list("asdfölkjqwermnbv"), 4, 200, False))


def play_qualification(args, competitors, n_competitor_pairings):
    for p in generatePairings(
        competitors,
        n_competitor_pairings=n_competitor_pairings,
    ):
        yield match(args=args, competitors=p)


def play_final(args, competitors, demo):
    scoreboard = match(args=args, competitors=competitors, demo=demo)
    return scoreboard


def tournament(competitors):
    args = parse()
    args.passive = True

    assert len(competitors) == len(set(competitors)), "Duplicate competitors"

    # Qualification
    scoreboards = play_qualification(
        args=args,
        competitors=competitors,
        n_competitor_pairings=2,
    )
    gui = QualificationGUI(None)
    for intermediate_result in aggregateScoreboards(scoreboards):
        gui.render_leaderboard(sorted(intermediate_result.items(), key=lambda x: x[1], reverse=True))
        logger.info(f"Intermediate qualification result: {intermediate_result}")
    gui.quit()
    
    qualification_results = dict(
        sorted(intermediate_result.items(), key=lambda x: x[1], reverse=True)
    )
    logger.info(f"Qualification results: {qualification_results}")
    with open(LOG_PATH / "qualification_results.txt", 'w') as file:
        file.write(json.dumps(qualification_results))

    # Playoffs
    with open(LOG_PATH / "qualification_results.txt", 'r') as file:
        qualification_results = json.loads(file.read())
    playoff_competitors = ["assignment_session_1." + k for k in qualification_results.keys()][3:3 + PAIRING_CARDINALITY]
    playoff_results = play_final(args=args, competitors=playoff_competitors, demo=False)
    playoff_results= dict(
        sorted(playoff_results.items(), key=lambda x: x[1], reverse=True)
    )
    logger.info(f"Playoff results: {playoff_results}")
    with open(LOG_PATH / "playoff_results.txt", 'w') as file:
        file.write(json.dumps(playoff_results))

    # Final
    with open(LOG_PATH / "qualification_results.txt", 'r') as file:
        playoff_results = json.loads(file.read())
    with open(LOG_PATH / "playoff_results.txt", 'r') as file:
        playoff_results = json.loads(file.read())
    final_competitors = ["assignment_session_1." + k for k in qualification_results.keys()][:PAIRING_CARDINALITY - 1]
    final_competitors.append(["assignment_session_1." + k for k in playoff_results.keys()][0])
    final_results = play_final(args=args, competitors=final_competitors, demo=True)
    logger.info(f"Final results: {final_results}")
    with open(LOG_PATH / "final_results.txt", 'w') as file:
        file.write(json.dumps(final_results))


if __name__ == "__main__":
    logging.basicConfig(filename=LOG_PATH / 'tournament.log', level=logging.INFO)
    logger.info('Started')
    tournament(competitors=[
        "rule_based_agent",
        "coin_collector_agent",
        "peaceful_agent",
        "random_agent",
    ])
    

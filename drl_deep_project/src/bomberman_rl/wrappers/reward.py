from gymnasium import Wrapper
from functools import reduce

from ..envs import settings as s
from ..envs import events as e


class ScoreRewardWrapper(Wrapper):
    """
    This example reward is mostly based on the actual score.

    Note that you can not use this Gymnasium Wrapper Interface during the tournament!
    (Because every agent must act on the same environment.)

    You can use this to kickstart your learning experiments, though.
    """

    rewards = {
        e.KILLED_OPPONENT: s.REWARD_KILL,
        e.COIN_COLLECTED: s.REWARD_COIN,
        e.KILLED_SELF: s.REWARD_KILL_SELF,
        # e.SURVIVED_ROUND: .1,
        # e.GOT_KILLED: -5
    }
    
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        new_state, _, terminated, truncated, info = super().step(action)
        reward = reduce(lambda r, e: r + self.rewards.get(e, 0), info["events"], 0)
        self.current_state = new_state
        return new_state, reward, terminated, truncated, info
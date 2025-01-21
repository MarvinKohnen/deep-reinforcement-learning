from gymnasium import ObservationWrapper
from gymnasium.wrappers import FlattenObservation
from copy import deepcopy
import numpy as np
                

class RestrictedKeysWrapper(ObservationWrapper):
    """
    This example wrapper restricts the observation state space.

    Note that you can not use this Gymnasium Wrapper Interface during the tournament!
    (Because every agent must act on the same environment.)

    You can use this to kickstart your learning experiments, though.
    """
    def __init__(self, env, keys):
        super().__init__(env)
        self.keys = keys
        self.observation_space = deepcopy(self.observation_space)
        for k in set(self.observation_space.spaces.keys()) - set(self.keys):
            self.observation_space.spaces.pop(k)

    def observation(self, obs):
        if obs is None:
            return None
        else:
            for k in set(obs.keys()) - set(self.keys):
                obs.pop(k)
            return obs


class FlattenWrapper(FlattenObservation):
    """
    This example wrapper flattens the observation state space from multiple dict entries in np.array format to a single np.array.

    Note that you can not use this Gymnasium Wrapper Interface during the tournament!
    (Because every agent must act on the same environment.)

    You can use this to kickstart your learning experiments, though.
    """

    def observation(self, obs):
        if obs is None:
            return None
        else:
            return super().observation(obs)
        

class FixedLengthOpponentsInfo(ObservationWrapper):
    """
    Returns a fixed length Sequence for state entry `` in order for VecEnv to work.
    This is fixed via wrapper in order to not make a last minute change to the raw state interface for tournament participants.
    """
    def __init__(self, env, n_opponents):
        super().__init__(env)
        self.n_opponents = n_opponents

    def observation(self, obs):
        if obs is None:
            return None
        else:
            n_deads = self.n_opponents - len(obs["opponents_info"])
            walls = obs["walls"]
            dead_opponent = {
                "score": 0,
                "bombs_left": 0,
                "position": np.zeros_like(walls)
            }
            obs["opponents_info"] += (dead_opponent,) * n_deads
            return obs
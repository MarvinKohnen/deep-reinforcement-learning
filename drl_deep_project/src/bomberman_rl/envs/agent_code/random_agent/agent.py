import numpy as np

from ...actions import ActionSpace
from ..interface import RuleBasedAgent

class Agent(RuleBasedAgent):
    def setup(self):
        self.rng = np.random.default_rng()

    def act(self, *args, **kwargs):
        return ActionSpace.sample()
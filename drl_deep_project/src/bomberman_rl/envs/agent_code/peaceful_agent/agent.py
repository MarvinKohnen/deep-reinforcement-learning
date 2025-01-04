import numpy as np

from ...actions import ActionSpace, Actions
from ..interface import RuleBasedAgent

class Agent(RuleBasedAgent):
    def setup(self):
        self.rng = np.random.default_rng()

    def act(self, *args, **kwargs):
        action = Actions.BOMB.value
        while action == Actions.BOMB.value:
            action = ActionSpace.sample()
        return action
import numpy as np

from bomberman_rl import RuleBasedAgent, ActionSpace

class Agent(RuleBasedAgent):
    """
    Sticking to the ``RuleBasedAgent`` interface is mandatory.
    It enables your agent to **play** as proper part of the environment (``/src/bomberman_rl/envs/agent_code/<agent>``) which is required by our Tournament.
    (Demonstration only - do not inherit)
    """
    def setup(self):
        self.rng = np.random.default_rng()

    def act(self, *args, **kwargs):
        return ActionSpace.sample()
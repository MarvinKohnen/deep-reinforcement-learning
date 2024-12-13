from ..interface import RuleBasedAgent

class Agent(RuleBasedAgent):
    """
    This class is technically required by the environment!
    """
    def act(self, *args, **kwargs):
        return kwargs["env_user_action"]
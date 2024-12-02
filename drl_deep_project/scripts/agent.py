class RuleBaseAgent:
    """
    Stick to this interface to enable later competition.
    (Demonstration only - do not inherit)
    """
    def __init__(self):
        self.setup()

    def setup(self):
        """
        Before episode. Use this to setup action related state that is required to act on the environment.
        """
        pass

    def act(self, state: dict) -> int:
        """
        Before step. Return action based on state.

        :param state: The state of the environment.
        """
        raise NotImplementedError()


class RLAgent(RuleBaseAgent):
    """
    An agent that wants to learn can profit from further Callbacks.
    (Demonstration only - do not inherit)
    """
    def __init__(self):
        super().__init__()
        self.setup_training()

    def setup_training(self):
        """
        Before episode (optional). Use this to setup additional learning related state e.g. a replay memory, hyper parameters etc.
        """
        pass

    def game_events_occurred(
        self,
        old_state: dict,
        self_action: str,
        new_state: dict,
        events: list[str],
    ):
        """
        After step in environment (optional). Use this e.g. for model training.

        :param old_state: Old state of the environment.
        :param self_action: Performed action.
        :param new_state: New state of the environment.
        :param events: Events that occurred during step. These might be used for Reward Shaping.
        """
        pass

    def end_of_round(self):
        """
        After episode ended (optional). Use this e.g. for model training and saving.
        """
        pass

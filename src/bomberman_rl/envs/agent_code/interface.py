from logging import Logger

class RuleBasedAgent:
    """
    Stick to this interface to enable later competition.
    (Demonstration only - do not inherit)
    """
    def __init__(self):
        self.logger = Logger("Agent")
        self.setup()

    def setup(self):
        """
        Before episode (optional). Use this to setup action related state that is required to act on the environment.
        """
        pass

    def act(self, state, **kwargs) -> int:
        """
        Before step. Return action based on state.

        :param state: The state of the environment.
        """
        raise NotImplementedError()


class LearningAgent(RuleBasedAgent):
    """
    A learning agent (located within the environment package) profits from further callbacks.
    If you manually operate on the environment, like from ``/scripts``, it is not strictly necessary to follow this interface.
    However, the example training loop in main.py supports this interface by calling the respective callbacks.
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
        old_state,
        self_action,
        new_state,
        events,
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
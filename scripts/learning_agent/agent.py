from bomberman_rl import LearningAgent, events as e

from .q_learning import Model

# Custom events
SCORE_INCREASED = "SCORE_INCREASED"

class Agent(LearningAgent):
    """
    Sticking to the ``LearningAgent`` interface is optional.
    It enables your agent to **learn** as proper part of the environment (``/src/bomberman_rl/envs/agent_code/<agent>``) in order to enable Self-Play.
    The example training loop in main.py supports this interface as well by calling the respective callbacks.
    (Demonstration only - do not inherit)
    """
    def __init__(self):
        self.setup()
        self.setup_training()

    def setup(self):
        """
        Before episode. Use this to setup action related state that is required to act on the environment.
        """
        self.q_learning = Model()

    def act(self, state, **kwargs) -> int:
        """
        Before step. Return action based on state.

        :param state: The state of the environment.
        """
        return self.q_learning.act(state)[0].item()

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
        custom_events = self._custom_events(old_state, new_state)
        reward = self._shape_reward(events + custom_events)
        self.q_learning.experience(old_state=old_state, action=self_action, new_state=new_state, reward=reward)
        self.q_learning.optimize_incremental()


    def end_of_round(self):
        """
        After episode ended (optional). Use this e.g. for model training and saving.
        """
        self.q_learning.optimize_incremental()
        self.q_learning.save_weights() # save model in case this was last round


    def _custom_events(self, old_state, new_state):
        """
        Just an idea to demonstrate that you are not solely bound to official events for reward shaping
        """
        custom_events = []
        if "score" in old_state and old_state["score"] < new_state["score"]:  # does not trigger due to current observation wrapper in main.py
            custom_events.append(SCORE_INCREASED)
        return custom_events

    def _shape_reward(self, events: list[str]) -> float:
        """
        Shape rewards here instead of in an Environment Wrapper in order to be more flexible (e.g. use this agent as proper component of the environment where no environment wrappers are possible)
        """
        reward_mapping = {
            SCORE_INCREASED: 1, # does not trigger due to current observation wrapper in main.py
            e.COIN_COLLECTED: 1,
            e.GOT_KILLED: -1,
            e.KILLED_SELF: -1
        }
        return sum([reward_mapping.get(event, 0) for event in events])
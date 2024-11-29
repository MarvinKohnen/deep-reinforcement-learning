import pickle
from collections import deque, namedtuple
from typing import List

import events as e
import numpy as np

# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
BOMB_TRIGGERED = False


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.debug("setup training")
    print("setup training")
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # (s, a, r, s')
    self.transitions.append(
        Transition(
            old_game_state,
            self_action,
            reward_from_events(self, events),
            new_game_state,
        )
    )
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )
    self.model.start_of_action(new_game_state)
    if old_game_state:
        self.model.update()
        events = self.model.get_events(events, self_action)
        self.model.rewards = reward_from_events(self, events)

        self.model.q_learning(self_action)

        # state_to_features is defined in callbacks.py
        self.transitions.append(
            Transition(old_game_state, self_action, new_game_state, self.model.rewards)
        )
    self.model.end_of_action()
    if self.coin:
        with open("my-saved-coins-model.pt", "wb") as file:
            pickle.dump(self.model, file)
    if self.crate:
        with open("my-saved-crates-model.pt", "wb") as file:
            pickle.dump(self.model, file)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    old_game_state = self.transitions[-1][0]
    events = self.model.get_events(events, last_action)
    self.model.rewards = reward_from_events(self, events)

    self.model.q_learning(last_action)

    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )
    self.transitions.append(
        Transition(last_game_state, last_action, None, self.model.rewards)
    )
    self.epsilon -= self.epsilon_decay_value
    # Store the model
    if self.coin:
        with open("my-saved-coins-model.pt", "wb") as file:
            pickle.dump(self.model, file)
    if self.crate:
        with open("my-saved-crates-model.pt", "wb") as file:
            pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    if self.coin:
        game_rewards = {
            # coin rewards
            e.COIN_COLLECTED: 50,
            e.KILLED_OPPONENT: 5,
            e.INVALID_ACTION: -10,
            e.WAITED: -5,
            # idea: the custom event is bad
            FURTHER_COIN: -5,
            e.BOMB_DROPPED: -100,
        }

    if self.crate:
        game_rewards = {
            e.INVALID_ACTION: -50,
            # e.CRATE_DESTROYED: 5,
            e.BOMB_DROPPED: -20,
            DANGER: -20,
            CLOSER_BOMB: -50,
            WAIT_NO_BOMB: -5,
            FURTHER_CRATE: -20,
            PERFECT_MOVE: 20,
            NOT_PERFECT_MOVE: -10,
            e.GOT_KILLED: -100,
            e.KILLED_SELF: -10,
        }

    reward_sum = 0
    for event in np.unique(events):
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # print(reward_sum, events)
    return reward_sum

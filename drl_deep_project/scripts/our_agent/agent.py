import numpy as np
import torch
from bomberman_rl import LearningAgent, events as e

from .q_learning import Model

# Custom events
SCORE_INCREASED = "SCORE_INCREASED"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        state = self.get_scope_representation(state)
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
        # Process both states using get_scope_representation
        old_state_processed = self.get_scope_representation(old_state)
        new_state_processed = None if new_state is None else self.get_scope_representation(new_state)
        
        reward = self._shape_reward(events)
        self.q_learning.experience(
            old_state=old_state_processed,
            action=self_action,
            new_state=new_state_processed,
            reward=reward
        )
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

    def get_scope_representation(self, state):
        self_pos = np.argwhere(state["self_pos"] == 1)[0]
        walls = state["walls"]
        crates = state["crates"]
        coins = state["coins"]
        opponents = state["opponents_pos"]

        # Initialize reachable positions (position is marked as a 2) and danger map into one array
        danger_map = self.get_danger_map(state)
        # print("Danger map:\n", self.format_array(danger_map))
        scope_representation = danger_map
        if not scope_representation[self_pos[0], self_pos[1]] < 0:
            scope_representation[self_pos[0], self_pos[1]] = 2
        # Traverse the board and mark reachable positions using BFS
        # Mark 1 for reachable positions, 0 for unreachable positions, -10 for walls, -9 for crates, -1 for bombs and ??? explosions

        queue = [self_pos]
        while queue:
            pos = queue.pop(0)
            for d in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                # Get new position
                new_pos = (pos[0] + d[0], pos[1] + d[1])
                # Mark crates
                if crates[new_pos[0], new_pos[1]] == 1:
                    scope_representation[new_pos[0], new_pos[1]] = -9
                if (
                    coins[new_pos[0], new_pos[1]] == 1
                    and scope_representation[new_pos[0], new_pos[1]] >= 0
                ):
                    scope_representation[new_pos[0], new_pos[1]] = 10
                if (
                    opponents[new_pos[0], new_pos[1]] == 1
                    and scope_representation[new_pos[0], new_pos[1]] >= 0
                ):
                    scope_representation[new_pos[0], new_pos[1]] = -50

                # Mark reachable positions
                if scope_representation[new_pos[0], new_pos[1]] == 0:
                    scope_representation[new_pos[0], new_pos[1]] = 1
                    queue.append(new_pos)
        return torch.tensor(scope_representation, device=device, dtype=torch.float32).flatten()

    def get_danger_map(self, state):
        bombs = -1 * state["bombs"]
        walls = -10 * state["walls"]
        explosions = -20 * (state["explosions"] // 10)
        danger_map = bombs + walls + explosions

        for i in range(danger_map.shape[0]):
            for j in range(danger_map.shape[1]):
                """
                if (
                    danger_map[i, j] == -2
                    or danger_map[i, j] == -3
                    or danger_map[i, j] == -4
                ):"""
                if -5 < danger_map[i, j] < 0:
                    radius = abs(danger_map[i, j])

                    # danger_map[i, j] = -20
                    wall_hit_up = False
                    wall_hit_right = False
                    wall_hit_down = False
                    wall_hit_left = False

                    for k in range(1, radius):
                        if (
                            i + k < danger_map.shape[0]
                            and danger_map[i + k, j] != -10
                            and not wall_hit_up
                        ):
                            danger_map[i + k, j] = -20
                        if i + k < danger_map.shape[0] and danger_map[i + k, j] == -10:
                            wall_hit_up = True
                        if (
                            j + k < danger_map.shape[1]
                            and danger_map[i, j + k] != -10
                            and not wall_hit_right
                        ):
                            danger_map[i, j + k] = -20
                        if j + k < danger_map.shape[1] and danger_map[i, j + k] == -10:
                            wall_hit_right = True
                        if (
                            i - k >= 0
                            and danger_map[i - k, j] != -10
                            and not wall_hit_down
                        ):
                            danger_map[i - k, j] = -20
                        if i - k >= 0 and danger_map[i - k, j] == -10:
                            wall_hit_down = True
                        if (
                            j - k >= 0
                            and danger_map[i, j - k] != -10
                            and not wall_hit_left
                        ):
                            danger_map[i, j - k] = -20
                        if j - k >= 0 and danger_map[i, j - k] == -10:
                            wall_hit_left = True

        return danger_map



    
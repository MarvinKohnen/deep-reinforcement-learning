from bomberman_rl import Actions
import numpy as np

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
        # For tracking own position and history
        self.position = None
        self.bombs_left = 0
        self.score = 0
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT # cartesian coordinates representation
        self.possible_moves = [(-1, 0), (0, 1), (1, 0), (0, -1)] # UP, RIGHT, DOWN, LEFT # matrix coordinates representation

    def act(self, state: dict) -> int:
        """
        Before step. Return action based on state.

        :param state: The state of the environment.
        """
        position_indices = np.where(state["self_info"]["position"] == 1)
        self.position = (position_indices[0][0], position_indices[1][0])
        self.bombs_left = state["self_info"]["bombs_left"]
        self.score = state["self_info"]["score"]

        # Get all observations
        walls = state["walls"]
        crates = state["crates"]
        bombs = state["bombs"]
        explosions = state["explosions"]
        coins = state["coins"]
        
        # Get valid actions
        valid_actions = self.get_valid_actions(state)
        print("Valid actions:", [Actions(action).name for action in valid_actions])
        # Choose a random valid action
        action = np.random.choice(valid_actions)
        print("Chosen action:", Actions(action).name)
        return action

    def get_scope_representation(self, state):
        walls = state["walls"].transpose()
        crates = state["crates"].transpose()
        bombs = state["bombs"].transpose()
        explosions = state["explosions"].transpose()
        # Initialize reachable positions (self.position is marked as a 2)
        scope_representation = np.zeros(walls.shape)
        scope_representation[self.position[1], self.position[0]] = 2
        # Traverse the board and mark reachable positions using BFS
        # Mark 1 for reachable positions, 0 for unreachable positions, -1 for walls and crates, -2 for bombs and explosions
        queue = [self.position]
        while queue:
            pos = queue.pop(0)
            for d in self.directions:
                new_pos = (pos[0] + d[0], pos[1] + d[1])
                # Mark walls and crates
                if walls[new_pos[0], new_pos[1]] == 1 or crates[new_pos[0], new_pos[1]] == 1:
                    scope_representation[new_pos[0], new_pos[1]] = -1
                # Mark bombs and explosions
                elif bombs[new_pos[0], new_pos[1]] > 0 or explosions[new_pos[0], new_pos[1]] > 0:
                    scope_representation[new_pos[0], new_pos[1]] = -2
                # Mark reachable positions
                else:
                    if scope_representation[new_pos[0], new_pos[1]] == 0:
                        scope_representation[new_pos[0], new_pos[1]] = 1
                        queue.append(new_pos)
        return scope_representation.transpose()

    def get_valid_actions(self, state):
        walls = state["walls"]
        crates = state["crates"]
        bombs = state["bombs"]
        explosions = state["explosions"]
        # Check if bombs are left
        valid_actions = []
        if self.bombs_left > 0:
            valid_actions.append(Actions.BOMB.value)
        # Check if we can wait (not standing on bomb or explosion)
        if bombs[self.position[0], self.position[1]] == 0 and explosions[self.position[0], self.position[1]] == 0:
            valid_actions.append(Actions.WAIT.value)
        # Check where we can move
        print("self.position:", self.position)
        for d in self.directions:
            new_pos = (self.position[0] + d[0], self.position[1] + d[1])
            print("Using direction:", d, "corresponding to Action:", Actions(self.directions.index(d)).name, "to get new position:", new_pos)
            if walls[new_pos[0], new_pos[1]] == 0 \
            and crates[new_pos[0], new_pos[1]] == 0 \
            and bombs[new_pos[0], new_pos[1]] == 0 \
            and explosions[new_pos[0], new_pos[1]] == 0:
                valid_actions.append(self.directions.index(d))
        return valid_actions

    def format_array(self, array):
        return "\n".join(" ".join(f"{int(item):2}" for item in array[0]) if i == 0 else " " + " ".join(f"{int(item):2}" for item in row) for i, row in enumerate(array))



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

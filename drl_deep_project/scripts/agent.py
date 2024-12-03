from bomberman_rl import Actions
import numpy as np


class RuleBasedAgent:
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
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT

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

        # print("Bombs:\n", self.format_array(bombs))
        # print("Explosions:\n", self.format_array(explosions))

        # Get scope representation
        scope_representation = self.get_scope_representation(state)
        print(
            "Scope representation:\n",
            self.format_array(scope_representation.transpose()),
        )

        # Get valid actions
        valid_actions = self.get_valid_actions(state, scope_representation)
        print("Valid actions:", [Actions(action).name for action in valid_actions])

        # Choose a random valid action
        if valid_actions == []:
            valid_actions.append(Actions.WAIT.value)
        action = np.random.choice(valid_actions)
        print("Chosen action:", Actions(action).name)
        return action

    def get_scope_representation(self, state):
        walls = state["walls"]
        crates = state["crates"]
        coins = state["coins"]
        opponents = state["opponents_pos"]

        # Initialize reachable positions (self.position is marked as a 2) and danger map into one array
        danger_map = self.get_danger_map(state)
        # print("Danger map:\n", self.format_array(danger_map))
        scope_representation = danger_map
        if not scope_representation[self.position[0], self.position[1]] < 0:
            scope_representation[self.position[0], self.position[1]] = 2
        # Traverse the board and mark reachable positions using BFS
        # Mark 1 for reachable positions, 0 for unreachable positions, -10 for walls, -9 for crates, -1 for bombs and ??? explosions

        queue = [self.position]
        while queue:
            pos = queue.pop(0)
            for d in self.directions:
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
                # Mark reachable positions
                if scope_representation[new_pos[0], new_pos[1]] == 0:
                    scope_representation[new_pos[0], new_pos[1]] = 1
                    queue.append(new_pos)
        return scope_representation

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

    def get_valid_actions(self, state, scope_representation):
        walls = state["walls"]
        crates = state["crates"]
        bombs = state["bombs"]
        explosions = state["explosions"]
        # Check if bombs are left
        valid_actions = []

        # Check if we can wait (not standing on bomb or explosion)
        if scope_representation[self.position] > 0:
            valid_actions.append(Actions.WAIT.value)
        # Check where we can move
        # print("self.position:", self.position)
        for d in self.directions:
            new_pos = (self.position[0] + d[0], self.position[1] + d[1])
            # print("Using direction:", d, "corresponding to Action:", Actions(self.directions.index(d)).name, "to get new position:", new_pos)
            if scope_representation[new_pos[0], new_pos[1]] > 0:
                valid_actions.append(self.directions.index(d))

        coins_found, directions_to_the_money = self.check_for_coins(
            scope_representation
        )

        bomb_allowed, _ = self.check_squares(state, scope_representation)

        if self.bombs_left > 0 and bomb_allowed:
            return [Actions.BOMB.value]

        if scope_representation[self.position[0], self.position[1]] == -1:
            _, direction_escape = self.check_squares(state, scope_representation)
            for action in valid_actions.copy():
                print(f"Action in valid_actions: {action}")
                if action not in direction_escape:
                    valid_actions.remove(action)
                    print(f"removed the following action: {Actions(action).name}")

        print("Coins Found.")
        print(f"Direction to Coin: {directions_to_the_money}")
        if coins_found and scope_representation[self.position] > 0:
            for action in valid_actions.copy():
                if action not in directions_to_the_money:
                    valid_actions.remove(action)
                    print(f"removed the following action: {Actions(action).name}")

        return valid_actions

    def check_squares(self, state, scope_representation):
        # Simply implement pathfinding at this point lol
        bomb_allowed = False
        bomb_wish = False
        direction = []
        # 4 in horizontal oder vertikal frei
        if (
            scope_representation.shape[0] > self.position[0] + 4
            and scope_representation[self.position[0] + 4, self.position[1]] == 1
        ):
            print("found path horizontal: right")
            bomb_allowed = True
            direction.append(1)  # right

        if (
            scope_representation.shape[1] > self.position[1] + 4
            and scope_representation[self.position[0], self.position[1] + 4] == 1
        ):
            print("found path vertikal: down ")
            bomb_allowed = True
            direction.append(2)  # down

        if (
            0 < self.position[0] - 4
            and scope_representation[self.position[0] - 4, self.position[1]] == 1
        ):
            print("found path horizontal: left")
            bomb_allowed = True
            direction.append(3)  # left

        if (
            0 < self.position[1] - 4
            and scope_representation[self.position[0], self.position[1] - 4] == 1
        ):
            print("found path vertiakl: up")
            bomb_allowed = True
            direction.append(0)  # up

        # Diagonal frei?
        for i in range(
            max(0, self.position[0] - 4),
            min(self.position[0] + 4, scope_representation.shape[0]),
        ):
            for j in range(
                max(0, self.position[1] - 4),
                min(self.position[1] + 4, scope_representation.shape[1]),
            ):
                if (
                    i != self.position[0]
                    and j != self.position[1]
                    and scope_representation[i, j] == 1
                ):
                    bomb_allowed = True
                    delta_i = abs(i - self.position[0])
                    delta_j = abs(j - self.position[1])
                    if delta_i <= delta_j:
                        if self.position[1] < j:
                            print("found path eck")
                            direction.append(2)  # down
                        if self.position[1] > j:
                            print("found path eck")
                            direction.append(0)  # up
                    if delta_j <= delta_i:
                        if self.position[0] < i:
                            print("found path eck")
                            direction.append(1)  # right
                        if self.position[0] > i:
                            print("found path eck")
                            direction.append(3)  # left

        if (
            scope_representation.shape[0] > self.position[0] + 1
            and scope_representation[self.position[0] + 1, self.position[1]] == -9
            or scope_representation.shape[1] > self.position[1] + 1
            and scope_representation[self.position[0], self.position[1] + 1] == -9
            or 0 < self.position[0] - 1
            and scope_representation[self.position[0] - 1, self.position[1]] == -9
            or 0 < self.position[1] - 1
            and scope_representation[self.position[0], self.position[1] - 1] == -9
        ):
            bomb_wish = True

        return bomb_allowed and bomb_wish, list(set(direction))

    def check_for_coins(self, scope_representation):
        # Simply implement pathfinding at this point lol

        coins_found = False
        direction = []

        for i in range(scope_representation.shape[0]):
            for j in range(scope_representation.shape[1]):
                if scope_representation[i, j] == 10:
                    coins_found = True

                    if self.position[1] < j:
                        direction.append(2)  # down
                    if self.position[1] > j:
                        direction.append(0)  # up
                    if self.position[0] < i:
                        direction.append(1)  # right
                    if self.position[0] > i:
                        direction.append(3)  # left

        return coins_found, list(set(direction))

    def format_array(self, array):
        # Calculate the width needed to align all numbers correctly
        max_width = max(len(str(int(item))) for row in array for item in row)

        return "\n".join(
            " ".join(f"{int(item):{max_width}}" for item in array[0])
            if i == 0
            else " " + " ".join(f"{int(item):{max_width}}" for item in row)
            for i, row in enumerate(array)
        )


class RLAgent(RuleBasedAgent):
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

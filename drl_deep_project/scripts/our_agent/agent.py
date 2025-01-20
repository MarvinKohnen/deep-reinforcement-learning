import numpy as np
import torch
from bomberman_rl import LearningAgent, events as e
import time
from pathlib import Path
from bomberman_rl.envs.settings import ROWS, COLS

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
    def __init__(self, weights=None):
        # Define reward mapping as class attribute
        self.reward_mapping = {
            e.COIN_COLLECTED: 5,
            e.INVALID_ACTION: -0.5,
            e.KILLED_OPPONENT: 5,
            e.CRATE_DESTROYED: 1,
            e.COIN_FOUND: 1,
            e.GOT_KILLED: -5,
            e.WAITED: -0.1,
        }
        self.q_learning = Model(weights_suffix=weights)
        #define variables to control map flipping
        self.flipUD = 0
        self.flipLR = 0
        self.setup()
        self.setup_training()

    def setup(self):
        """
        Before episode. Use this to setup action related state that is required to act on the environment.
        """
        self.q_learning = Model()

    def transform_state(self, state):
        self.update_transformation(state)
        for key in state:
            if key == 'round' or key == 'step':
                continue
            elif key == 'self_info':
                state[key]['position'] = self.transform_map(state[key]['position'])
            elif key == 'opponents_info':
                for op in range(len(state[key])):
                    state[key][op]['position'] = self.transform_map(state[key][op]['position'])
            else:
                state[key] = self.transform_map(state[key])

    def transform_map(self, map):
        #print(obs)
        #print("flipud:", self.flipUD)
        #print("fliplr:", self.flipLR)
        if self.flipUD == 1:
            map = np.array(np.flipud(map))
        if self.flipLR == 1:
            map = np.array(np.fliplr(map))
        return map

    def transform_action(self, action):
        if self.flipUD == 1 and (action == 2 or action == 0):
            return abs(action-2)
        if self.flipLR == 1 and (action == 1 or action == 3):
            return 1 if action == 3 else 3
        return action

    def update_transformation(self, state):
        self.flipUD = 0
        self.flipLR = 0
        pos = np.argwhere(state["self_pos"] == 1)[0]
        #print("pos: ", pos)
        height = len(state['self_pos'])
        width = len(state['self_pos'][0])
        if pos[0] > (height+1)//2 - 1:
            self.flipUD = 1
        if pos[1] > (width+1)//2 - 1:
            self.flipLR = 1

    def act(self, state, **kwargs) -> int:
        """
        Process state directly using all information except self_info and opponents_info
        """
        # Determine if we're in evaluation mode (not training)
        eval_mode = not kwargs.get('train', True)

        # Update transformation variables
        self.transform_state(state)
        # Convert relevant state components to numpy arrays
        # state_elements = [
        #     np.array(state['walls']).flatten(),
        #     np.array(state['crates']).flatten(),
        #     np.array(state['coins']).flatten(),
        #     np.array(state['bombs']).flatten(),
        #     np.array(state['explosions']).flatten(),
        #     np.array(state['self_pos']).flatten(),
        #     np.array(state['opponents_pos']).flatten(),
        #     np.array([state['self_info']['bombs_left']])  # Wrap single value in list
        # ]

        # Concatenate all elements into a single numpy array
        # state_array = np.concatenate(state_elements)

        #state_array = self.get_optimized_state(state)
        state_array = self.get_scope_representation(state)

        # Convert numpy array to tensor
        state_tensor = torch.tensor(state_array, device=device, dtype=torch.float32)
        #state_tensor = self.get_scope_representation(state)

        out_action = self.q_learning.act(state_tensor, eval_mode=eval_mode)[0].item()

        #print("action b4 transform:", out_action)
        #print("action after transform:", self.transform_action(out_action))

        return self.transform_action(self.q_learning.act(state_tensor, eval_mode=eval_mode)[0].item())

    def setup_training(self):
        """
        Before episode (optional). Use this to setup additional learning related state.
        """
        if self.q_learning.load:
            # Try to get timestamp from existing model weights
            model_files = list(Path("scripts/our_agent/models").glob("dqn_*.pt"))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                self.training_timestamp = latest_model.name.replace('dqn_', '').replace('.pt', '')
            else:
                self.training_timestamp = time.strftime("%Y%m%d_%H%M%S")
        else:
            # Fresh model gets new timestamp
            self.training_timestamp = time.strftime("%Y%m%d_%H%M%S")


    def game_events_occurred(self, old_state, self_action, new_state, events):
        """
        After step in environment. Use this for model training.
        """
        #Transform both states
        self.transform_state(old_state)
        if new_state is not None: self.transform_state(new_state)

        # Process both states
        old_state_tensor = torch.tensor(self.get_scope_representation(old_state), device=device, dtype=torch.float32)
        new_state_tensor = None if new_state is None else torch.tensor(self.get_scope_representation(new_state),
                                                                       device=device, dtype=torch.float32)

        reward = self._shape_reward(events)

        self.q_learning.experience(
            old_state=old_state_tensor,
            action=self_action,
            new_state=new_state_tensor,
            reward=reward
        )
        return self.q_learning.optimize_incremental()
         
    def end_of_round(self):
        """
        After episode ended (optional). Use this e.g. for model training and saving.
        """
        self.q_learning.optimize_incremental()
        self.q_learning.save_weights()

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
        Shape rewards using the reward mapping defined in __init__
        """
        return sum([self.reward_mapping.get(event, 0) for event in events])

    def get_scope_representation(self, state):
        # Get state information
        self_pos = np.argwhere(state["self_pos"] == 1)[0]
        walls = state["walls"]
        crates = state["crates"]
        coins = state["coins"]
        opponents = state["opponents_pos"]

        # Define representation values
        wall_value = -1
        crate_value = 2
        coin_value = 10
        self_value = 5
        reachable_value = 1
        opponent_value = -10
        # Initialize reachable positions (position is marked as a 2) and danger map into one array
        danger_map = self.get_danger_map(state)
        scope_representation = danger_map
        if not scope_representation[self_pos[0], self_pos[1]] < 0:
            scope_representation[self_pos[0], self_pos[1]] = self_value

        # Traverse the board and mark reachable positions using BFS
        queue = [self_pos]
        while queue:
            pos = queue.pop(0)
            for d in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                # Get new position
                new_pos = (pos[0] + d[0], pos[1] + d[1])
                # Mark crates
                if crates[new_pos[0], new_pos[1]] == 1:
                    scope_representation[new_pos[0], new_pos[1]] = crate_value
                if (
                    coins[new_pos[0], new_pos[1]] == 1
                    and scope_representation[new_pos[0], new_pos[1]] >= 0
                ):
                    scope_representation[new_pos[0], new_pos[1]] = coin_value
                if (
                    opponents[new_pos[0], new_pos[1]] == 1
                    and scope_representation[new_pos[0], new_pos[1]] >= 0
                ):
                    scope_representation[new_pos[0], new_pos[1]] = opponent_value

                # Mark reachable positions
                if scope_representation[new_pos[0], new_pos[1]] == 0:
                    scope_representation[new_pos[0], new_pos[1]] = reachable_value
                    queue.append(new_pos)


        # Create fixed window around agent 
        window_size = 4  # This gives 7x7 (3 cells in each direction)
        padded_scope = np.pad(
            scope_representation,
            window_size,
            mode='constant',
            constant_values=wall_value   # Use wall value to indicate out-of-bounds as padding
        )

        # For window_size = 3 and agent at position (x,y) in scope_representation:
        agent_x, agent_y = self_pos[0] + window_size, self_pos[1] + window_size  # (x+3,y+3) because we added padding
        window = padded_scope[
            agent_x - window_size : agent_x + window_size + 1, # (x+3)-3 : (x+3)+3+1 -> [x:x+7]
            agent_y - window_size : agent_y + window_size + 1  # (y+3)-3 : (y+3)+3+1 -> [y:y+7]
        ]
        return window.flatten() #torch.tensor(window, device=device, dtype=torch.float32).flatten()

    def get_danger_map(self, state):
        bombs = -1 * state["bombs"]
        walls = -1 * state["walls"]
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

    def get_reward_mapping(self):
        """Return the reward mapping"""
        return self.reward_mapping

    def get_optimized_state(self, state):
        """
        Returns essential features using the proper state format from state_space.py
        """
        # Get agent position from self_pos array
        x, y = np.argwhere(state['self_pos'] == 1)[0]

        # 1. Immediate surroundings (8 features)
        surroundings = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS:
                # Check if position is walkable (no wall, no crate)
                is_wall = state['walls'][nx,ny] == 1
                is_crate = state['crates'][nx,ny] == 1
                surroundings.append(1.0 if not (is_wall or is_crate) else 0.0)
            else:
                surroundings.append(0.0)

        # 2. Closest objects in each direction (16 features)
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        direction_features = []
        max_look_ahead = min(4, max(ROWS, COLS) // 2)

        for dx, dy in directions:
            features = [0.0, 0.0, 0.0, 0.0]  # coin, crate, bomb, enemy
            for distance in range(1, max_look_ahead + 1):
                nx, ny = x + dx * distance, y + dy * distance
                if 0 <= nx < ROWS and 0 <= ny < COLS:
                    if state['coins'][nx,ny] == 1:
                        features[0] = 1.0 - (distance-1)/max_look_ahead
                    if state['crates'][nx,ny] == 1:
                        features[1] = 1.0 - (distance-1)/max_look_ahead
                    if state['bombs'][nx,ny] > 0:  # bombs have timer values
                        features[2] = 1.0 - (distance-1)/max_look_ahead
                    if state['opponents_pos'][nx,ny] == 1:
                        features[3] = 1.0 - (distance-1)/max_look_ahead
            direction_features.extend(features)

        # 3. Global features (8 features)
        max_distance = ROWS + COLS

        # Find nearest coin
        coin_positions = np.where(state['coins'] == 1)
        coin_distances = [self.manhattan_distance((x,y), (cx,cy))
                        for cx, cy in zip(*coin_positions)] if len(coin_positions[0]) > 0 else []
        nearest_coin = min(coin_distances) if coin_distances else max_distance

        # Find nearest bomb
        bomb_positions = np.where(state['bombs'] > 0)
        bomb_distances = [self.manhattan_distance((x,y), (bx,by))
                        for bx, by in zip(*bomb_positions)] if len(bomb_positions[0]) > 0 else []
        nearest_bomb = min(bomb_distances) if bomb_distances else max_distance

        # Find nearest enemy
        enemy_positions = np.where(state['opponents_pos'] == 1)
        enemy_distances = [self.manhattan_distance((x,y), (ex,ey))
                        for ex, ey in zip(*enemy_positions)] if len(enemy_positions[0]) > 0 else []
        nearest_enemy = min(enemy_distances) if enemy_distances else max_distance

        global_features = [
            nearest_coin/max_distance,
            nearest_bomb/max_distance,
            nearest_enemy/max_distance,
            np.sum(state['coins'])/9,       # total coins remaining
            np.sum(state['bombs'] > 0)/4,   # total active bombs
            np.sum(state['opponents_pos'])/3,  # total opponents
            float(state['self_info']['bombs_left']),  # can place bomb
            float(np.any(state['explosions'] > 0))    # explosions present
        ]

        # Combine all features (32 total)
        final_state = np.array(surroundings + direction_features + global_features)

        # print("\n=== DQN Input Features ===")
        # print("Total features:", len(final_state))
        # print("Surroundings (8):", surroundings)
        # print("Direction features (16):", direction_features)
        # print("Global features (8):", global_features)
        # print("Complete state:", final_state)
        # print("========================\n")

        return final_state

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two points"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
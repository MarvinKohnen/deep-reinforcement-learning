import numpy as np
import torch
from bomberman_rl import LearningAgent, events as e
import time

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
#        self.pos = (0,0)
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
                for op in key:
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
        # Update transformation variables
        self.transform_state(state)

        # Convert relevant state components to numpy arrays
        state_elements = [
            np.array(state['walls']).flatten(),
            np.array(state['crates']).flatten(),
            np.array(state['coins']).flatten(),
            np.array(state['bombs']).flatten(),
            np.array(state['explosions']).flatten(),
            np.array(state['self_pos']).flatten(),
            np.array(state['opponents_pos']).flatten(),
            np.array([state['self_info']['bombs_left']])
        ]

        # Concatenate all elements into a single numpy array
        state_array = np.concatenate(state_elements)

        # Convert numpy array to tensor
        state_tensor = torch.tensor(state_array, device=device, dtype=torch.float32)

        return self.transform_action(self.q_learning.act(state_tensor)[0].item())

    def setup_training(self):
        """
        Before episode (optional). Use this to setup additional learning related state e.g. a replay memory, hyper parameters etc.
        """
        # Create a timestamp for this training run
        self.training_timestamp = time.strftime("%Y%m%d_%H%M%S")
        pass


    def game_events_occurred(self, old_state, self_action, new_state, events):
        """
        After step in environment. Use this for model training.
        """

        self.transform_state(old_state)
        self.transform_state(new_state)

        def state_to_tensor(state):
            if state is None:
                return None
            
            state_elements = [
                np.array(state['walls']).flatten(),
                np.array(state['crates']).flatten(),
                np.array(state['coins']).flatten(),
                np.array(state['bombs']).flatten(),
                np.array(state['explosions']).flatten(),
                np.array(state['self_pos']).flatten(),
                np.array(state['opponents_pos']).flatten(),
                np.array([state['self_info']['bombs_left']])  # Wrap single value in list
            ]
            
            state_array = np.concatenate(state_elements)
            return torch.tensor(state_array, device=device, dtype=torch.float32)

        # Process both states
        old_state_tensor = state_to_tensor(old_state)
        new_state_tensor = state_to_tensor(new_state)

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
        # Save model with timestamp
        self.q_learning.save_weights(suffix=self.training_timestamp)

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
            e.COIN_COLLECTED: 1,
            e.INVALID_ACTION: -0.5,
            e.KILLED_OPPONENT: 5,
            e.CRATE_DESTROYED: 0.1,
            e.GOT_KILLED: -5,
            e.WAITED: -0.1,
            #e.MOVED_LEFT: -0.1,      
            #e.MOVED_RIGHT: -0.1,   
            #e.MOVED_UP: -0.1,
            #e.MOVED_DOWN: -0.1,
        }
        return sum([reward_mapping.get(event, 0) for event in events])

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
        window_size = 3  # This gives 7x7 (3 cells in each direction)
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
        return torch.tensor(window, device=device, dtype=torch.float32).flatten()

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

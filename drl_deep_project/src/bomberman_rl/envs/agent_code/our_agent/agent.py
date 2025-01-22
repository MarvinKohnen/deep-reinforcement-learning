import numpy as np
import torch
from bomberman_rl import LearningAgent, events as e
import time
from pathlib import Path
from bomberman_rl.envs.settings import ROWS, COLS

from .q_learning import Model as SingleDQN
from .double_q_learning import Model as DoubleDQN

# Custom events
SCORE_INCREASED = "SCORE_INCREASED"

device = torch.device("cpu")

class Agent(LearningAgent):
    """
    Sticking to the ``LearningAgent`` interface is optional.
    It enables your agent to **learn** as proper part of the environment (``/src/bomberman_rl/envs/agent_code/<agent>``) in order to enable Self-Play.
    The example training loop in main.py supports this interface as well by calling the respective callbacks.
    (Demonstration only - do not inherit)
    """
    def __init__(self, weights="20250121_212644", use_double_dqn=False):
        # Define reward mapping as class attribute
        self.reward_mapping = {
            e.COIN_COLLECTED:2,
            e.INVALID_ACTION: -0.5,
            e.KILLED_OPPONENT: 5,
            e.CRATE_DESTROYED: 1,
            e.COIN_FOUND: 1,
            e.GOT_KILLED: -5,
            e.WAITED: -0.1,
        }
        self.use_double_dqn = use_double_dqn
        self.setup(weights)
        self.setup_training()

    def setup(self, weights):
        """
        Before episode. Use this to setup action related state that is required to act on the environment.
        """
        ModelClass = DoubleDQN if self.use_double_dqn else SingleDQN
        self.q_learning = ModelClass(weights_suffix=weights)


    def act(self, state, **kwargs) -> int:
        """
        Process state directly using enhanced state representation
        """
        eval_mode = not kwargs.get('train', True)
        print(eval_mode)
        
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

        state_array = self.get_enhanced_state(state)

        # Convert numpy array to tensor
        state_tensor = torch.tensor(state_array, device=device, dtype=torch.float32)
        
        return self.q_learning.act(state_tensor, eval_mode=eval_mode)[0].item()

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
        # Process both states
        old_state_tensor = self.get_enhanced_state(old_state)
        new_state_tensor = None if new_state is None else self.get_enhanced_state(new_state)

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
    
    def get_enhanced_state(self, state):
        """
        Enhanced state representation (72 features) for larger network (72->128->64->6)
        """
        # Get agent position from self_pos array
        x, y = np.argwhere(state['self_pos'] == 1)[0]
        
        # 1. Immediate surroundings (8 features)
        surroundings = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS:
                is_wall = state['walls'][nx,ny] == 1
                is_crate = state['crates'][nx,ny] == 1
                surroundings.append(1.0 if not (is_wall or is_crate) else 0.0)
            else:
                surroundings.append(0.0)
        
        # 2. Danger awareness in 8 directions (40 features)
        danger_features = []
        blast_range = 3
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            direction_danger = [0.0] * 5  # [immediate_explosion, bomb_timer1, bomb_timer2, bomb_timer3, escape_route]
            max_range = blast_range if dx*dy == 0 else 2  # Shorter range for diagonals
            
            has_escape = False
            for distance in range(1, max_range + 2):
                nx, ny = x + dx * distance, y + dy * distance
                if 0 <= nx < ROWS and 0 <= ny < COLS:
                    if state['walls'][nx,ny] == 1:
                        break
                    if state['explosions'][nx,ny] > 2:
                        direction_danger[0] = 1.0 - (distance-1)/max_range
                    if state['bombs'][nx,ny] > 0:
                        timer = state['bombs'][nx,ny]
                        direction_danger[int(timer)] = 1.0 - (distance-1)/max_range
                    if not (state['walls'][nx,ny] == 1 or state['crates'][nx,ny] == 1 or 
                        state['bombs'][nx,ny] > 0 or state['explosions'][nx,ny] > 2):
                        has_escape = True

            direction_danger[4] = 1.0 if has_escape else 0.0
            danger_features.extend(direction_danger)

        
        # 3. Object awareness in 8 directions (24 features)
        object_features = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            features = [0.0, 0.0, 0.0]  # [coin_distance, crate_distance, clear_path]
            max_look = 4 if dx*dy == 0 else 3  # Adjust look range for diagonals
            
            for distance in range(1, max_look + 1):
                nx, ny = x + dx * distance, y + dy * distance
                if 0 <= nx < ROWS and 0 <= ny < COLS:
                    if state['coins'][nx,ny] == 1 and features[0] == 0:
                        features[0] = 1.0 - (distance-1)/max_look
                    if state['crates'][nx,ny] == 1 and features[1] == 0:
                        features[1] = 1.0 - (distance-1)/max_look
                    if not (state['walls'][nx,ny] == 1 or state['crates'][nx,ny] == 1):
                        features[2] = 1.0
            object_features.extend(features)
        
        # 4. Global features (8 features)
        max_distance = ROWS + COLS
        coin_positions = np.where(state['coins'] == 1)
        coin_distances = [self.manhattan_distance((x,y), (cx,cy)) 
                        for cx, cy in zip(*coin_positions)] if len(coin_positions[0]) > 0 else []
        nearest_coin = min(coin_distances) if coin_distances else max_distance
        
        danger_count = 0
        for dx in range(-blast_range, blast_range + 1):
            for dy in range(-blast_range, blast_range + 1):
                if dx == 0 or dy == 0:  # Only in blast directions
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < ROWS and 0 <= ny < COLS:
                        if state['bombs'][nx,ny] > 0 or state['explosions'][nx,ny] > 2:
                            danger_count += 1
        
        global_features = [
            nearest_coin/max_distance,
            danger_count/(4 * blast_range),
            float(state['self_info']['bombs_left']),
            float(np.any(state['explosions'] > 2)),
            np.sum(state['bombs'] > 0)/4,
            np.sum(state['coins'])/9,
            float(self.in_danger(state)),
            float(np.any(state['bombs'] == 1))
        ]
        
        # Combine all features (72 total)
        final_state = np.array(surroundings + danger_features + object_features + global_features)
        return final_state

    def in_danger(self, state):
        """Check if agent is in immediate danger from bombs or explosions"""
        x, y = np.argwhere(state['self_pos'] == 1)[0]
        blast_range = 3

        # Check if on explosion
        if state['explosions'][x, y] > 0:
            return True

        # Check for bombs in blast range
        for dx in range(-blast_range, blast_range + 1):
            for dy in range(-blast_range, blast_range + 1):
                if dx == 0 or dy == 0:  # Only check in blast directions
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < ROWS and 0 <= ny < COLS:
                        if state['bombs'][nx, ny] > 0:
                            # Check if there's a wall between agent and bomb
                            if not any(state['walls'][x + i*np.sign(dx), y + i*np.sign(dy)] == 1 
                                     for i in range(1, abs(dx) + abs(dy))):
                                return True
        return False
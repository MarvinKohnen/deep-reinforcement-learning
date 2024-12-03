import numpy as np
from bomberman_rl import Actions

class EliasWorkingAgent:
    def __init__(self):
        """Initialize the agent"""
        self.setup()
        self.possible_moves = [(-1, 0, Actions.LEFT.value), (1, 0, Actions.RIGHT.value), (0, -1, Actions.UP.value), (0, 1, Actions.DOWN.value)]

    def setup(self):
        """"Setup the agent"""
        self.rng = np.random.default_rng()

    def act(self, state: dict) -> int:
        """
        Choose an action based on the current state of the game
        
        This is dona via a heirarchy of rules:
        1. If a bomb is near, get away from it (if possible, otherwise wait)
        2. If a coin is near, get it
        3. If a crate is near, lay a bomb
        4. If an enemy is near, lay a bomb
        5. Otherwise, move randomly (if possible, otherwise wait)
        
        """
        position = state['self_pos']
        x, y = np.argwhere(position == 1)[0]

        # get away from bombs
        if self.bomb_near(state, x, y):
            safe_action = self.find_safe_move(state, x, y)
            if safe_action is not None:
                return safe_action
            else:
                # we are stuck
                return Actions.WAIT.value

        # if coin near, get it
        coin_action = self.move_towards(state, x, y)
        if coin_action is not None:
            return coin_action

        # lay bomb to destroy crate
        if state['self_info']['bombs_left'] > 0 and self.adjacent_to_crate(state, x, y):
            return Actions.BOMB.value
        
        # if enemy near, lay bomb
        if self.enemy_near(state, x, y):
            return Actions.BOMB.value

        # move randomly
        random_action = self.random_move(state, x, y)
        if random_action is not None:
            return random_action
        else:
            # we are stuck
            return Actions.WAIT.value
        

    def bomb_near(self, state, x, y):
        danger_map = state['bombs'] + state['explosions']
        return danger_map[x, y] > 0 # check at current position

    def enemy_near(self, state, x, y):
        """Check if an enemy is near"""
        for score, bombs_left, pos in state['opponents_info']:
            pos_x, pos_y = np.where(np.array(state['opponents_info'][0]['position']) == 1)
            if np.abs(pos_x - x) + np.abs(pos_y - y) <= 1:
                return True
        return False

    def find_safe_move(self, state, x, y):
        """Find a safe move to get away from bombs"""
        # check all possible directions
        for delta_x, delta_y, action in self.possible_moves:
            if self.is_safe(state, x + delta_x, y + delta_y): # if safe then move to that direction
                return action
        return None

    def is_safe(self, state, x, y):
        """Check if a position is safe (or even possible) to move to"""
        # extract relevant information from the state
        crates = state['crates']
        walls = state['walls']
        bombs = state['bombs']
        explosions = state['explosions']
        
        # check if position is within the map
        if not (0 <= x < walls.shape[0] and 0 <= y < walls.shape[1]): return False
        # check if there are crates, walls, bombs, or explosions
        return (crates[x, y] == 0 and walls[x, y] == 0 and bombs[x, y] == 0 and explosions[x, y] == 0)

    def move_towards(self, state, x, y):
        """Move towards the nearest coin"""
        # get all positions of coins
        target_positions = np.argwhere(state['coins'] > 0)
        if target_positions.size == 0: return None
        # find the nearest coin
        nearest = target_positions[np.argmin(np.abs(target_positions - np.array([x, y])).sum(axis=1))]
        delta_x, delta_y = nearest - np.array([x, y])
        # move towards the nearest coin if possible
        if abs(delta_x) > abs(delta_y):
            if delta_x > 0 and self.is_safe(state, x + 1, y):
                return Actions.RIGHT.value
            elif delta_x < 0 and self.is_safe(state, x - 1, y):
                return Actions.LEFT.value
        else:
            if delta_y > 0 and self.is_safe(state, x, y + 1):
                return Actions.DOWN.value
            elif delta_y < 0 and self.is_safe(state, x, y - 1):
                return Actions.UP.value
        return None

    def adjacent_to_crate(self, state, x, y):
        """Check if the agent is adjacent to a crate"""
        crates = state['crates']
        # check all possible directions
        for delta_x, delta_y, _ in self.possible_moves:
            new_x, new_y = x + delta_x, y + delta_y
            # check if position is within the map and if there is a crate
            if 0 <= new_x < crates.shape[0] and 0 <= new_y < crates.shape[1]:
                if crates[new_x, new_y] == 1: return True
        return False
    
    def random_move(self, state, x, y):
        """Choose a random move that is safe"""
        safe_moves = []
        # check for all possible directions if they are safe
        for delta_x, delta_y, action in self.possible_moves:
            new_x, new_y = x + delta_x, y + delta_y
            if self.is_safe(state, new_x, new_y):
                safe_moves.append(action)
        # return a random safe move (if there are any)
        if safe_moves:
            return self.rng.choice(safe_moves)
        return None
import numpy as np

from bomberman_rl import Actions


class RandomAgent:
    def __init__(self):
        self.setup()

    def setup(self):
        self.rng = np.random.default_rng()

    def act(self, state: dict) -> int:
        action = Actions.BOMB.value
        while action == Actions.BOMB.value:
            action = np.argmax(self.rng.random(len(Actions)))

        pos = state['self_info']['position']
        print(np.argmax(pos))
        print(np.argmax(pos)//17, np.argmax(pos)%17)

        true_pos = (np.argmax(pos)//17, np.argmax(pos)%17)
        #get coordinates for all bombs, repeat for every bomb
        x, y = 0, 0
        blast_coords = [(x, y)]

        for i in range(1, 3 + 1):
            if state['walls'][x + i, y] == 1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, 3 + 1):
            if state['walls'][x + i, y] == 1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, 3 + 1):
            if state['walls'][x + i, y] == 1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, 3 + 1):
            if state['walls'][x + i, y] == 1:
                break
            blast_coords.append((x, y - i))

        if true_pos in blast_coords:
            pass #move away from bomb (and maybe towards coin)

        return action

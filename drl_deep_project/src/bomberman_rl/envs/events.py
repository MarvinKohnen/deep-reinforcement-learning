# Events maintained by the environment
# i.e. calculated and returned as part of the 'information' dict as part of the return of environment.step()

MOVED_LEFT = "MOVED_LEFT"
MOVED_RIGHT = "MOVED_RIGHT"
MOVED_UP = "MOVED_UP"
MOVED_DOWN = "MOVED_DOWN"
WAITED = "WAITED"
INVALID_ACTION = "INVALID_ACTION"

BOMB_DROPPED = "BOMB_DROPPED"
BOMB_EXPLODED = "BOMB_EXPLODED"

CRATE_DESTROYED = "CRATE_DESTROYED"
COIN_FOUND = "COIN_FOUND"
COIN_COLLECTED = "COIN_COLLECTED"

KILLED_OPPONENT = "KILLED_OPPONENT"
KILLED_SELF = "KILLED_SELF"

GOT_KILLED = "GOT_KILLED"
OPPONENT_ELIMINATED = "OPPONENT_ELIMINATED"
SURVIVED_ROUND = "SURVIVED_ROUND"

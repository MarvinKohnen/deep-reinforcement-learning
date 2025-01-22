import numpy as np
import json
from gymnasium.spaces import Space, Discrete, MultiBinary, MultiDiscrete, Sequence, Dict, Text

from . import settings as s


def _multi_discrete_space(n=1):
    """
    Arena shaped space
    """
    if n == 1:
        return MultiBinary([s.COLS, s.ROWS])
    else:
        return MultiDiscrete(np.ones((s.COLS, s.ROWS)) * n)

def observation_space():
    SInt = Discrete(2 ** 20)
    SWalls = _multi_discrete_space()
    SCrates = _multi_discrete_space()
    SCoins = _multi_discrete_space()
    SBombs = _multi_discrete_space(s.BOMB_TIMER + 1) # 0 = no bomb
    SExplosions = _multi_discrete_space(15)
    SAgentPos = _multi_discrete_space()
    SOpponentsPos = _multi_discrete_space()
    SAgent = Dict({
        "score": SInt,
        "bombs_left": Discrete(2),
        "position": _multi_discrete_space()
    })
    SOpponents = Sequence(SAgent)
    return Dict({
        "round": SInt,
        "step": SInt,
        "walls": SWalls,
        "crates": SCrates,
        "coins": SCoins,
        "bombs": SBombs,
        "explosions": SExplosions,
        "self_pos": SAgentPos,
        "opponents_pos": SOpponentsPos,
        "self_info": SAgent,
        "opponents_info": SOpponents
    })


def legacy2gym(state):

    def _agent_legacy2gym(agent, pos):
        return {
            "score": agent[1],
            "bombs_left": int(agent[2]),
            "position": pos
        }
    
    if state is None:
        return None
    
    walls = (state["field"] == - 1).astype("int16")
    crates = (state["field"] == 1).astype("int16")

    coins = np.zeros(state["field"].shape, dtype="int16")
    if len(state["coins"]):
        coins[*zip(*state["coins"])] = 1

    bombs = np.zeros(state["field"].shape, dtype="int16")
    if len(state["bombs"]):
        pos, timer = zip(*state["bombs"])
        pos = list(pos)
        timer_feature = s.BOMB_TIMER - np.array(list(timer))
        bombs[*zip(*pos)] = timer_feature

    self_pos = np.zeros(state["field"].shape, dtype="int16")
    _, _, _, pos = state["self"]
    self_pos[*pos] = 1

    opponents_pos = np.zeros(state["field"].shape, dtype="int16")
    if len(state["others"]):
        positions = [pos for _, _, _, pos in state["others"]]
        opponents_pos[*zip(*positions)] = 1

    self_info = _agent_legacy2gym(state["self"], self_pos)
    
    single_opponents_pos = []
    for _, _, _, pos in state["others"]:
        single_opponent_pos = np.zeros(state["field"].shape, dtype="int16")
        single_opponent_pos[*pos] = 1
        single_opponents_pos.append(single_opponent_pos)
    opponents_info = tuple([_agent_legacy2gym(agent, pos) for agent, pos in zip(state["others"], single_opponents_pos)])

    return {
        "round": state["round"],
        "step": state["step"],
        "walls": walls,
        "crates": crates,
        "coins": coins,
        "bombs": bombs,
        "explosions": state["explosion_map"],
        "self_pos": self_pos,
        "opponents_pos": opponents_pos,
        "self_info": self_info,
        "opponents_info": opponents_info
    }


def gym2legacy(state):
    field = state["crates"].copy()
    field[state["walls"] == 1] = -1

    self_pos = list(zip(*np.where(state["self_pos"] == 1)))[0]
    self_info = ("", state["self_info"]["score"], state["self_info"]["bombs_left"], self_pos)

    opponents = []
    for o in state["opponents_info"]:
        o_pos = list(zip(*np.where(o["position"] == 1)))[0]
        opponents.append(("", o["score"], o["bombs_left"], o_pos))

    bombs = []
    for x, y in zip(*np.nonzero(state["bombs"])):
        bombs.append(((x, y), s.BOMB_TIMER - state["bombs"][x][y]))

    coins = list(zip(*np.where(state["coins"] == 1)))

    return {
        "round": state["round"],
        "step": state["step"],
        "field": field,
        "self": self_info,
        "others": opponents,
        "bombs": bombs,
        "coins": coins,
        "explosion_map": state["explosions"]
    }


def serializeGym(state):

    def _serializeLiteral(literal):
        if isinstance(literal, dict):
            return {
                k: _serializeLiteral(v) for k, v in literal.items()
            }
        elif isinstance(literal, np.ndarray):
            return literal.tolist()
        elif isinstance(literal, list):
            return [_serializeLiteral(l) for l in literal]
        elif isinstance(literal, tuple):
            return tuple(_serializeLiteral(l) for l in literal)
        else:
            return literal

    state = {
        k: _serializeLiteral(v) for k, v in state.items()
    }
    return json.dumps(state)


def deserializeGym(state):

    def _deserializeArray(arr):
        return np.array(arr, dtype=np.int16)
    
    state = json.loads(state)
    for k in ["walls", "crates", "coins", "bombs", "explosions", "self_pos", "opponents_pos"]:
        state[k] = _deserializeArray(state[k])
    state["self_info"]["position"] = _deserializeArray(state["self_info"]["position"])
    for opp in state["opponents_info"]:
        opp["position"] = _deserializeArray(opp["position"])

    return state
# Bomberman Reinforcement Learning

**Disclaimer: This code is still under development and subject to change.**

<img src="./docs/resources/icon.png" alt="Bomberman" width="250"/>

## Acknowledgements

This project was [originally developed](https://github.com/ukoethe/bomberman_rl) at the chair of Prof. Dr. Koethe at the university of Heidelberg.

## Description

Student project to train and compete Reinforcement Learning Agents in a Bomberman Environemnt.

## Getting Started

### Prerequisites

- Ideally conda

### Installation

1. Clone the repository:
   ```bash
   git clone https://zivgitlab.uni-muenster.de/ai-systems/bomberman_rl.git
   cd bomberman_rl
   ```

2. Create conda env:
   ```bash
   conda env create --name <name>
   conda activate <name>
   ```
   This will implicitly install this package as dependency in editable mode.

3. Alternative:
   ```bash
   pip install -e .
   ```
   Manually install further requirements.

### Run
- Watch arbitrary agents play
   ```bash
   python scripts/main.py --players rule_based_agent rule_based_agent
   ```
- Play yourself (movement: `Up`, `Down`, `Left`, `Right`; bomb: `Space`, wait: `Enter`)
   ```bash
   python scripts/main.py --players rule_based_agent rule_based_agent --user-play
   ```
- Further
    ```bash
   python scripts/main.py -h
   ```

## Versioning and Tags
Each version is identified by a tag in the format:
v*Major.Minor.Patch* with
- *Major*: Exercise Sheet number
- *Minor*: Feature additions
- *Patch*: Bug Fixes

See the [Changelog](./CHANGELOG.md) for which changes the tags refer to

### Check out a specific version
```bash
$ git checkout <tag>
```

## Develop
This package provides
- `src/`: bomberman *Environment* as Gymnasium environment
- `scripts/`: example *Agent* acting on the environment

### Environment
- Action space:
```python
class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4
    BOMB = 5
```
- Observation space:
```python
{
    'round': int,
    'step': int,
    'walls': np.array((17, 17), dtype=int16),
    'crates': np.array((17, 17), dtype=int16),
    'coins': np.array((17, 17), dtype=int16),
    'bombs': np.array((17, 17), dtype=int16),
    'explosions': np.array((17, 17), dtype=int16),
    'self_pos': np.array((17, 17), dtype=int16),
    'opponents_pos': np.array((17, 17), dtype=int16),
    'self_info': {
        'score': int,
        'bombs_left': int,
        'position': np.array((17, 17), dtype=int16)
    }
    'opponents_info': ({...}, {...}, {...})
}
```

### Agent
- We are talking **Rule Based** for now
    - you might stumble upon learning related code, though
- You can adapt agent and action loop, but you might wanna **stick to the interface**
    - this allows for later competition
    - see the `RuleBasedAgent` in `scripts/agent.py`
    - see the example implementation of a random agent in `scripts/random_agent/agent.py`

```python
class RandomAgent:
    def __init__(self):
        self.setup()

    def setup(self):
        self.rng = np.random.default_rng()

    def act(self, state: dict, **kwargs) -> int:
        action = Actions.BOMB.value
        while action == Actions.BOMB.value:
            action = np.argmax(self.rng.random(len(Actions)))
        return action
```

## Troubleshooting
The following  arm64 Macs, the given conda environment does not work.
```bash
conda create --name <name> python=3.11.10
conda activate <name>
pip install configargparse
pip install gymnasium
pip install pygame
PYTHONPATH=src python scripts/main.py
```
Manual installation of further packages.
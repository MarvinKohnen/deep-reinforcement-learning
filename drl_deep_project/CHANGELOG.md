
## [v2.2.0]
### Changed
- add curriculum scenarios 
   ```bash
   python scripts/main.py --players --train --scenario curriculum-coin
   python scripts/main.py --players --train --scenario curriculum-crate
   ```

## [v2.1.0]
### Changed
- change coordinate center from Top-Left to Bottom-Left: this switches the effect of the ```UP``` and ```DOWN``` actions

## [v2.0.1]
### Fixed
- clean entry point (main loop)
- start position of ```single-coin``` scenarios
- custom arena error handling

# [v2.0.0] December 6, 2024
### Added
- run a tournament
- single coin scenario, see ```settings.py```
### Changed
- command line arguments:
    - e.g. add ```--passive``` mode
    - for further changes, see ```--help```
- negative score for killing oneself
- interface of environment internal agents to equal environment external agent
### Fixed
- prevent PyGame window from rendering for render modes other than ```"human"```
- action format: environment only accepts action inputs from proper action space (i.e. numbers)
- enable custom avatars
- index out of range during rendering of arenas that are not quadratic

## [v1.0.1]
### Fixed
- ```Coins``` observation was missing in ```README.md``` (credit to student for pointing out)


# [v1.0.0] November 22, 2024
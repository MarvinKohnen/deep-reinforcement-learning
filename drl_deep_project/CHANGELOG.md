## [v2.0.0] Dec 6th
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
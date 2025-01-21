### Basic Observations

Inspired by AlphaGo/AlphaGo Zero

- The agent's position in the map does not affect the possible actions.
- The agent's position in the map does not significantly affect the optimal action.
- Example:
  - The agent is in the top left quadrant.
  - The optimal action is to move to the right.
  - If the map was mirrored along its vertical axis (including the agent's position), the optimal action would therefore be to move to the left.
  - However, the state representation/observation is different in the mirrored case.
  - Even though the optimal action is virtually the same (just mirrored).
- Conclusion: If we can ensure that the agent is only learning on any one quadrant, and only observes states where it is in that same quadrant, variance is reduced.

### Idea
 
- We transform the state space to only portray the agent in the top left quadrant:
  - If the agent is in any other quadrant, the matrix (numpy array) is flipped (mirrored) to place the agent in the top left quadrant.
  - If the agent walks across the halfway line either to the right or the bottom, the transformation is adapted dynamically: The agent "thinks" that it "bounced off" at the halfway line.
  - The actions are transformed accordingly: left <-> right, up <-> down.
- In theory, this should cut the state space by 3/4, without losing any information:
  - The whole state is still represented (just mirrored).
  - As there is no information tied specifically to the quadrant the agent is in, the transformation has no information loss.
  - Overfitting should not be an issue, as **all** input states have the agent in the top left quadrant (because all observed states are transformed before input).

### Results

- Due to an implementation error that did not show up easily in debugging, the state transformation did not work correctly in the beginning, nullifying its results before the error was caught.
- In the few runs after fixing that error, no discernible effect could be seen. We believe this is due to multiple reasons:
  - Most of the refined state abstractions used do not take much of the original state arrays into account immediately, instead using only the immediate surroundings and then calculating relative positions.
    - While this should still be stabilised by the transformation (e.g. the relative distance of the next wall to the left is likely in a narrower range), the effect is not as significant as when using e.g. the entire arrays for walls, bombs, etc.
    - Input features such as the distance to the nearest coin is not at all affected by the transformation, as the distance between 2 points in the map is invariant to any transformations.
  - Even though all observations have the agent in the top left quadrant, this might still lead to overfitting in some way, especially with the more refined state abstractions and the relative positions.
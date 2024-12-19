# Markov Decision Process

---

**Group Members:**
- Alina Micke (515183)
- Björn Balkow (514317)
- Mathis Hunke (508192)

**Files**
1. `grid_value_function.ipynb`: Code and Solutions for Exercise 2.1
2. `gymansium_env`  
   2.1. `grid_world_simulation.py`: File to create, register, and record the simulation of the grid world.  
   2.2. `cart_pole_simulation.py`: File to load and record the cart pole environment.
   2.3. `envs`: Custom Gymnasium Environments  
        2.3.1. `grid_world.py`: Custom GridWorld implementation for the Gymnasium Library.
3. `cart_pole_controller.ipynb`: Code and Solutions for Exercise 2.3
4. `grid-world-agent-recordings`: Video files showing the recordings of two episodes: one for few and one for many steps of the agent.
5. `cart-pole-agent-recordings`: Video files showing the individual episodes of the agent as well as their composition (`cart-pole-4-episode-sim.mp4`)
6. PDF: Description and Plots of results.

**Dependencies**
- numpy = "^2.1.2"
- pandas = "^2.2.3"
- scipy = "^1.14.1"
- jupyter = "^1.1.1"
- matplotlib = "^3.9.2"
- tqdm = "^4.66.5"
- gymnasium = {extras = ["other"], version = "^1.0.0"}
- swig = "^4.2.1"
- copier = "^9.4.1"
- seaborn = "^0.13.2"
- ipympl = "^0.9.4"
- ipywidgets = "^8.1.5"

Python Version: 3.13

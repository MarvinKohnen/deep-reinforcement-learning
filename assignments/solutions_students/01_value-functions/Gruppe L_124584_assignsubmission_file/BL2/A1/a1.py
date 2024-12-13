import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import seaborn as sns

class State:
    def __init__(self, coords:tuple, transitionDict:dict, value:float, discount:float, customTransitions:dict = None) -> None:
        self._coords = coords
        self._transitionDict = transitionDict #(0,0) : (node, reward, probability, value) 
        self._value = value
        self._discount = discount
        self.pre_calc_transitions(customTransitions)

    def __str__(self):
        tmpDict = {}
        for key in self._transitionDict:
            tmpDict[key] = (self._transitionDict[key][0]._coords, self._transitionDict[key][1], self._transitionDict[key][2])
        return f"State: {self._coords}"
    
    #Calculates one step of the state value function (considering the given state transitions)
    def calc_value(self):
        newVal = 0
        for key in self._transitionDict:
            transitionData = self._transitionDict[key]
            nextState = transitionData[0]
            reward = transitionData[1]
            probability = transitionData[2]
            newVal += probability * (reward + self._discount * nextState._value)
        return newVal
    
    def calc_action_value_dependent(self):
        for key in self._transitionDict:
            transitionData = self._transitionDict[key]
            nextState = transitionData[0]
            nextStateVal = nextState._value
            reward = transitionData[1]
            newVal = reward + self._discount * nextStateVal
            tmp = list(transitionData)
            tmp[3] = newVal
            self._transitionDict[key] = tuple(tmp)
    
    def calc_action_value_indipendent(self):
        new_vals = []
        for key in self._transitionDict:
            transitionData = self._transitionDict[key]
            nextState = transitionData[0]
            reward = transitionData[1]
            sum_of_next_state_action_vals = 0
            for key_1 in nextState._transitionDict:
                transitionData_1 = nextState._transitionDict[key_1]
                policy = transitionData_1[2]
                val = transitionData_1[3]
                sum_of_next_state_action_vals += (policy*val)
            new_val = reward + self._discount * sum_of_next_state_action_vals
            new_vals.append(new_val)
        return new_vals
        



    #When initializing a state, set arbitrary (or custom) state transitions (will be corrected after all states are initialized)
    def pre_calc_transitions(self, customTransitions=None):
        row = self._coords[0]
        col = self._coords[1]

        if customTransitions != None:
            self._transitionDict = customTransitions
        else:
            self._transitionDict = {
                (row, col-1) : (None, 0, 0.25, 0),
                (row+1, col) : (None, 0, 0.25, 0),
                (row, col+1) : (None, 0, 0.25, 0),
                (row-1, col) : (None, 0, 0.25, 0)
            }
    
    #Update either the value of the given state or the state transitions
    def update(self, value = None, stateList = None, gridSize = None, action_values = None):
        #Update state transitions
        if not stateList is None:
            for key in copy(self._transitionDict):
                row = key[0]
                col = key[1]
                #Check if state is on the border of the grid
                if np.sign(row) == -1 or np.sign(col) == -1 or row > gridSize[0]-1 or col > gridSize[1]-1:
                    #If there is a transition to itself, delete the wrong transition and add to the probability of the transition to itself
                    if self._coords in self._transitionDict.keys():
                        self._transitionDict.pop(key)
                        tmp = list(self._transitionDict[self._coords])
                        tmp[2] += 0.25
                        self._transitionDict[self._coords] = tuple(tmp)
                    #Delete wrong transition and add a transition to itself
                    else:
                        self._transitionDict.pop(key)
                        self._transitionDict[self._coords] = (self, -1, 0.25, 0)
                #Insert the state object of the given transition into the transition
                else:
                    tmp = list(self._transitionDict[key])
                    tmp[0] = stateList[row][col]
                    self._transitionDict[key] = tuple(tmp)
            return len(list(self._transitionDict.keys()))
        #Update value of the given state
        elif value != None:
            self._value = value
        elif not action_values is None:
            for i, key in enumerate(self._transitionDict):
                tmp = list(self._transitionDict[key])
                tmp[3] = action_values[i]
                self._transitionDict[key] = tuple(tmp)
        else:
            raise("Either input Value or statelist not both")
    
class Grid:
    def __init__(self, rows:int, columns:int, discount:float, initValue:float, customTransitions:dict) -> None:
        self._size = (rows, columns)
        self._states = np.empty((rows, columns), dtype=State)
        self._discount = discount
        self._num_transitions = 0
        #Respective to the given size, create new States and possibly override specific state transitions
        for row in range(rows):
            for column in range(columns):
                customTransitionsForState = None
                #If a custom transition for a state is given, pass it to the new State
                if (row,column) in list(customTransitions.keys()):
                    customTransitionsForState = customTransitions[(row,column)]
                newState = State(coords=(row, column), 
                                 transitionDict={}, 
                                 value=initValue, 
                                 discount=discount,
                                 customTransitions=customTransitionsForState)
                self._states[row][column] = newState

        # Create transitions in states
        for state in self._states.flatten():
            self._num_transitions += state.update(stateList=self._states, gridSize = self._size)

    #Perform one iteration step in the calculation of the state values
    def step_state_action_value(self):
        newVals_state = np.zeros((self._size[0], self._size[1]))
        newVals_action = []
        #For every state calulate the new value and save it for return
        for i, row in enumerate(self._states):
            for o, state in enumerate(row):
                newVals_state[i][o] = state.calc_value()
                state.calc_action_value()
                for idx, key in enumerate(state._transitionDict):
                    transitionData = state._transitionDict[key]
                    newVals_action.append(transitionData[3])
                    
        #Update every state with their respective new value
        for state in self._states.flatten():
            val = newVals_state[state._coords[0]][state._coords[1]]
            state.update(value = val)

        return newVals_state, np.array(newVals_action)

    def step_state_value(self):
        new_vals = np.zeros((self._size[0], self._size[1]))
        #For every state calulate the new value and save it for return
        for i, row in enumerate(self._states):
            for o, state in enumerate(row):
                new_vals[i][o] = state.calc_value()
                    
        #Update every state with their respective new value
        for state in self._states.flatten():
            val = new_vals[state._coords[0]][state._coords[1]]
            state.update(value = val)
        return new_vals

    def step_action_value(self):
        new_vals = []
        for state in self._states.flatten():
            new_vals_state = state.calc_action_value_indipendent()
            new_vals.append(new_vals_state)
        
        for i, state in enumerate(self._states.flatten()):
            state.update(action_values=new_vals[i])
        
        new_vals = sum(new_vals, [])
        return np.array(new_vals)








if __name__ == "__main__":
    

    #Build grid
    customTransitions = {
        (0,1) : {(4,1) : (None, 10, 1.0, 0)},
        (0,3) : {(2,3) : (None, 5, 1.0, 0)}
    }
    startValue = 0
    discount = 0.9
    grid = Grid(5, 5, discount, startValue, customTransitions=customTransitions)

    #Initialize Value collection
    state_values_over_time = []
    t_0_state_vals = np.full((grid._size[0], grid._size[1]), 0.0, dtype=float)
    state_values_over_time.append(t_0_state_vals)

    action_values_over_time = []
    t_0_action_vals = np.zeros(grid._num_transitions)
    action_values_over_time.append(t_0_action_vals)

    #Grid iteration loop for state value calculation
    print("Computing State-Value...")
    try:
        from alive_progress import alive_bar
        iteration = 0
        with alive_bar(22) as bar:
            while discount ** iteration >= 0.1:
                #new_state_vals, new_action_vals = grid.step_state_action_value()
                new_state_vals = grid.step_state_value()
                new_action_vals = grid.step_action_value()
                state_values_over_time.append(new_state_vals)
                action_values_over_time.append(new_action_vals)
                iteration += 1
                bar()
    except:
        iteration = 0
        while discount ** iteration >= 0.1:
            #new_state_vals, new_action_vals = grid.step_state_action_value()
            new_state_vals = grid.step_state_value()
            new_action_vals = grid.step_action_value()
            state_values_over_time.append(new_state_vals)
            action_values_over_time.append(new_action_vals)
            iteration += 1

    avg_change_state_values = []
    error_state_values = []

    avg_change_action_values = []
    error_action_values = []

    x_ticks = []
    x_data = []
    for i, state_vals in enumerate(state_values_over_time):
        if not i == len(state_values_over_time)-1:
            x_ticks.append(f"{i}-{i+1}")
            x_data.append(i)

            change_state_vals = state_values_over_time[i+1] - state_values_over_time[i]
            avg_change_state_values.append(np.average(change_state_vals))
            error_state_values.append(np.std(change_state_vals))

            change_action_vals = action_values_over_time[i+1] - action_values_over_time[i]
            avg_change_action_values.append(np.average(change_action_vals))
            error_action_values.append(np.std(change_action_vals))

    x_data = np.array(x_data)
    
    width = 0.2
    barwidth = 0.125

    plt.grid(zorder=0)
    plt.bar(x_data, avg_change_state_values, width=barwidth, color="#0004e3", label="Average change of state value", zorder=3)
    plt.errorbar(x_data, avg_change_state_values, error_state_values, fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
    plt.xticks(x_data+width/2, x_ticks)

    plt.bar(x_data+width, avg_change_action_values, width=barwidth, color="#23e60e", label="Average change of action value", zorder=3)
    plt.errorbar(x_data+width, avg_change_action_values, error_action_values, fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
    plt.xticks(x_data+width/2, x_ticks)

    plt.legend(loc="lower right")
    plt.xlabel("Timesteps")
    plt.ylabel("Average change")
    plt.title("Convergence behaviour of state- and action-value function")

    plt.show()
    plt.close()






    
        


        

# code is taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import math
import random
from pathlib import Path
from collections import namedtuple, deque
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os 

from bomberman_rl import ActionSpace

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Array to store the tree
        self.data = np.zeros(capacity, dtype=object)  # Array to store the data/transitions
        self.write = 0  # Current writing position
        self.n_entries = 0  # Number of entries in tree

    def _propagate(self, idx, change):
        """Propagate the priority update up through the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find the index of the leaf with a given priority value"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return the total priority"""
        return self.tree[0]

    def add(self, priority, data):
        """Add new data to the tree"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        """Update the priority of a leaf"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Get a transition based on a priority value"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.eps = 0.01  # Small constant to ensure non-zero priority
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4  # Initial importance sampling weight
        self.beta_increment = 0.001  # Beta increment per sampling
        self.max_priority = 1.0  # Maximum priority for new transitions

    def push(self, *args):
        """Save a transition with maximum priority"""
        transition = Transition(*args)
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        """Sample a batch of transitions based on their priorities"""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Increase beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # Normalize weights

        return batch, indices, torch.FloatTensor(is_weights).to(device)

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error.item()) + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries


class DQN(nn.Module):
    """ State approximation via Multi-Layer Perceptron """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """ Expects flattened state vector """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def get_architecture_info(self):
        """Return the model's architecture information"""
        return {
            'type': self.__class__.__name__,
            'input_size': self.layer1.in_features,
            'hidden_layers': [
                self.layer1.out_features,
                self.layer2.out_features
            ],
            'output_size': self.layer3.out_features
        }

class Tabular(nn.Module):
    """ State approximation via Multi-Layer Perceptron """
    def __init__(self, n_observations, n_actions):
        super(Tabular, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """ Expects flattened state vector """
        return self.layer1(x)
    
class Model():
    def __init__(self, load=True, path=Path(__file__).parent / "model.pt", weights_suffix=None):
        self.batch_size = 128 # self.batch_size is the number of transitions sampled from the replay buffer
        self.gamma = 0.99 # self.gamma is the discount factor
        self.eps_start = 0.9 # self.eps_start is the starting value of epsilon
        self.eps_end = 0.1 # self.eps_end is the final value of epsilon
        self.eps_decay = 50000 # self.eps_decay controls the rate of exponential decay of epsilon, higher means a slower decay
        self.tau = 0.005 # self.tau is the update rate of the target network
        self.lr = 1e-4 # self.lr is the learning rate of the ``AdamW`` optimizer
        self.gradient_clipping = 100
        self.steps = 0
        self.path = path
        self.load = load # load trained model or init from scratch
        self.n_actions = ActionSpace.n
        self.memory = ReplayMemory(10_000)
        self.policy_net_a = None
        self.policy_net_b = None
        self.target_net = None  # We still keep one target network
        self.weights_suffix = weights_suffix
        self.training_timestamp = time.strftime("%Y%m%d_%H%M%S")

    def lazy_init(self, observation):
        # only on first observation can we lazy initialize as we have no upfront information on the environment
        self.n_observations = len(observation)
        # Initialize two policy networks
        self.policy_net_a = DQN(self.n_observations, self.n_actions).to(device)
        self.policy_net_b = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        
        if self.load:
            try:
                self.load_weights(self.weights_suffix)
            except FileNotFoundError:
                pass
        
        self.target_net.load_state_dict(self.policy_net_a.state_dict())
        # Create optimizers for both networks
        self.optimizer_a = optim.AdamW(self.policy_net_a.parameters(), lr=self.lr, amsgrad=True)
        self.optimizer_b = optim.AdamW(self.policy_net_b.parameters(), lr=self.lr, amsgrad=True)

    def act(self, state, eval_mode=False):
        if self.policy_net_a is None:
            self.lazy_init(state)
        self.steps += 1
        state = state.clone().detach().to(device).unsqueeze(0)
        
        # Use eps_end during evaluation
        eps_threshold = self.eps_end if eval_mode else self.get_epsilon()
        
        if random.random() > eps_threshold:
            self.policy_net_a.eval()
            self.policy_net_b.eval()
            with torch.no_grad():
                q_values_a = self.policy_net_a(state)
                q_values_b = self.policy_net_b(state)
                average_q_values = (q_values_a + q_values_b) / 2
                return average_q_values.max(1).indices
        else:
            return torch.tensor([ActionSpace.sample()], device=device, dtype=torch.long)
        
    def optimize_incremental(self):
        """Double Q-learning optimization step with prioritized experience replay"""
        transitions, indices, is_weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).to(device)
        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.stack(batch.action).to(device)
        reward_batch = torch.stack(batch.reward).to(device).squeeze(1)

        # Randomly choose which network to update
        update_a = random.random() < 0.5
        policy_net = self.policy_net_a if update_a else self.policy_net_b
        other_net = self.policy_net_b if update_a else self.policy_net_a
        optimizer = self.optimizer_a if update_a else self.optimizer_b

        # Get current Q values
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute next state values using other network for action selection
        next_state_values = torch.zeros(state_batch.shape[0], device=device)
        if len(non_final_next_states) > 0:
            with torch.no_grad():
                # Use other network to select actions
                next_actions = other_net(non_final_next_states).max(1).indices.unsqueeze(1)
                # Use target network to evaluate those actions
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze()

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Calculate TD errors for priority updating
        td_errors = expected_state_action_values.unsqueeze(1) - state_action_values

        # Apply importance sampling weights to the loss
        criterion = nn.SmoothL1Loss(reduction='none')
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = (loss * is_weights.unsqueeze(1)).mean()

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), self.gradient_clipping)
        optimizer.step()

        # Update priorities in memory
        self.memory.update_priorities(indices, td_errors)

        self.update_target_net()
        return loss.item()

    def update_target_net(self):
        """
        Soft update of the target network's weights using average of both policy networks
        θ′ ← τ (θa + θb)/2 + (1 −τ )θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_a_dict = self.policy_net_a.state_dict()
        policy_net_b_dict = self.policy_net_b.state_dict()
        
        for key in target_net_state_dict:
            # Average the weights of both policy networks
            avg_weight = (policy_net_a_dict[key] + policy_net_b_dict[key]) / 2
            # Soft update using the average
            target_net_state_dict[key] = avg_weight * self.tau + target_net_state_dict[key] * (1 - self.tau)
        
        self.target_net.load_state_dict(target_net_state_dict)

    def experience(self, old_state, action, new_state, reward):
        """
        Save new experience
        """
        if self.policy_net_a is None:
            self.lazy_init(old_state)

        self.memory.push(
            torch.tensor(old_state, device=device, dtype=torch.float32),
            torch.tensor([action], device=device, dtype=torch.int64),
            None if new_state is None else torch.tensor(new_state, device=device, dtype=torch.float32),
            torch.tensor([reward], device=device, dtype=torch.float32))

    def save_weights(self):
        """Save both networks"""
        save_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "models"
        save_dir.mkdir(exist_ok=True)
        filename_a = f"dqn_a_{self.training_timestamp}.pt"
        filename_b = f"dqn_b_{self.training_timestamp}.pt"
        torch.save(self.policy_net_a.state_dict(), save_dir / filename_a)
        torch.save(self.policy_net_b.state_dict(), save_dir / filename_b)

    def load_weights(self, suffix=None):
        """Load both networks"""
        save_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "models"
        
        if suffix:
            filename_a = f"dqn_a_{suffix}.pt"
            filename_b = f"dqn_b_{suffix}.pt"
        else:
            model_files = list(save_dir.glob("dqn_a_*.pt"))
            if not model_files:
                print("No saved model found. Starting with fresh weights.")
                return
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            filename_a = latest_model.name
            filename_b = filename_a.replace('dqn_a_', 'dqn_b_')
        
        try:
            self.policy_net_a.load_state_dict(torch.load(save_dir / filename_a, weights_only=True))
            self.policy_net_b.load_state_dict(torch.load(save_dir / filename_b, weights_only=True))
            # Initialize target network with average of both networks
            target_state_dict = self.target_net.state_dict()
            for key in target_state_dict:
                target_state_dict[key] = (self.policy_net_a.state_dict()[key] + 
                                        self.policy_net_b.state_dict()[key]) / 2
            self.target_net.load_state_dict(target_state_dict)
            print(f"Successfully loaded weights from {filename_a} and {filename_b}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps / self.eps_decay)

    def get_hyperparameters(self):
        """Return the model's hyperparameters"""
        return {
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'eps_start': self.eps_start,
            'eps_end': self.eps_end,
            'eps_decay': self.eps_decay,
            'tau': self.tau,
            'lr': self.lr,
            'gradient_clipping': self.gradient_clipping,
            'memory_size': self.memory.capacity,
            # PER parameters
            'per_epsilon': self.memory.eps,
            'per_alpha': self.memory.alpha,
            'per_beta': self.memory.beta,
            'per_beta_increment': self.memory.beta_increment
        }

    def get_model_info(self):
        """Return model architecture information"""
        if self.policy_net_a is None:
            return {}
        return self.policy_net_a.get_architecture_info()
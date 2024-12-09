# code is taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import math
import random
from pathlib import Path
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from bomberman_rl import ActionSpace

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)

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
    def __init__(self, load=True, path=Path(__file__).parent / "model.pt"):
        self.batch_size = 128 # self.batch_size is the number of transitions sampled from the replay buffer
        self.gamma = 0.99 # self.gamma is the discount factor
        self.eps_start = 0.5 # self.eps_start is the starting value of epsilon
        self.eps_end = 0.05 # self.eps_end is the final value of epsilon
        self.eps_decay = 1000 # self.eps_decay controls the rate of exponential decay of epsilon, higher means a slower decay
        self.tau = 0.005 # self.tau is the update rate of the target network
        self.lr = 1e-4 # self.lr is the learning rate of the ``AdamW`` optimizer
        self.gradient_clipping = 100
        self.steps = 0
        self.path = path
        self.load = load # load trained model or init from scratch
        self.n_actions = ActionSpace.n
        self.memory = ReplayMemory(10_000)
        self.policy_net = None

    def lazy_init(self, observation):
        # only on first observation can we lazy initialize as we have no upfront information on the environment
        self.n_observations = len(observation)
        self.policy_net = Tabular(self.n_observations, self.n_actions).to(device)
        if self.load:
            try:
                self.load_weights()
            except FileNotFoundError:
                pass
        self.target_net = Tabular(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

    def act(self, state):
        if self.policy_net is None:
            self.lazy_init(state)
        self.steps += 1
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps / self.eps_decay)
        sample = random.random()
        if sample > eps_threshold:
            self.policy_net.eval()
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices
        else:
            return torch.tensor([ActionSpace.sample()], device=device, dtype=torch.long)
        
    def optimize_incremental(self):
        """
        One iteration of Q learning (Bellman optimality equation for Q values) on a random batch of past experience
        """
        self.policy_net.train()
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
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

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(state_batch.shape[0], device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values # max Q values of next state

        # Compute the optimal Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch # Bellman optimality equation

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.gradient_clipping)
        self.optimizer.step()

        self.update_target_net()

    def update_target_net(self):
        """
        Soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def experience(self, old_state, action, new_state, reward):
        """
        Save new experience
        """
        if self.policy_net is None:
            self.lazy_init(old_state)
        self.memory.push(
            torch.tensor(old_state, dtype=torch.float32),
            torch.tensor([action], device=device, dtype=torch.int64),
            None if new_state is None else torch.tensor(new_state, dtype=torch.float32),
            torch.tensor([reward], device=device, dtype=torch.float32))

    def save_weights(self):
        torch.save(self.policy_net.state_dict(), self.path)

    def load_weights(self):
        self.policy_net.load_state_dict(torch.load(self.path, weights_only=True))
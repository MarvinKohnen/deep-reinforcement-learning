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
        self.eps_decay = 5000 # self.eps_decay controls the rate of exponential decay of epsilon, higher means a slower decay
        self.tau = 0.005 # self.tau is the update rate of the target network
        self.lr = 1e-4 # self.lr is the learning rate of the ``AdamW`` optimizer
        self.gradient_clipping = 100
        self.steps = 0
        self.path = path
        self.load = load # load trained model or init from scratch
        self.n_actions = ActionSpace.n
        self.memory = ReplayMemory(10_000)
        self.policy_net = None
        self.weights_suffix = weights_suffix  # Store the source weights suffix
        self.training_timestamp = time.strftime("%Y%m%d_%H%M%S")  # New timestamp for this run

    def lazy_init(self, observation):
        # only on first observation can we lazy initialize as we have no upfront information on the environment
        self.n_observations = len(observation)
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)  # Create target_net first
        
        if self.load:
            try:
                self.load_weights(self.weights_suffix)  # Pass weights_suffix here
            except FileNotFoundError:
                pass
        
        # Load target net after policy net is loaded
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

    def act(self, state, eval_mode=False):
        if self.policy_net is None:
            self.lazy_init(state)
        self.steps += 1
        state = state.clone().detach().to(device).unsqueeze(0)
        
        # Always act greedily during evaluation
        if eval_mode:
            self.policy_net.eval()
            with torch.no_grad():
                return self.policy_net(state).max(1).indices
        
        # Epsilon-greedy during training
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps / self.eps_decay)
        sample = random.random()
        if sample > eps_threshold:
            self.policy_net.eval()
            with torch.no_grad():
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
        return loss.item()

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
        
        # Convert numpy arrays to tensors
        old_state_tensor = torch.tensor(old_state, device=device, dtype=torch.float32)
        new_state_tensor = None if new_state is None else torch.tensor(new_state, device=device, dtype=torch.float32)
        
        self.memory.push(
            old_state_tensor,
            torch.tensor([action], device=device, dtype=torch.int64),
            new_state_tensor,
            torch.tensor([reward], device=device, dtype=torch.float32))

    def save_weights(self):
        """Save model with new timestamp"""
        save_dir = Path("scripts/our_agent/models")
        save_dir.mkdir(exist_ok=True)
        filename = f"dqn_{self.training_timestamp}.pt"
        torch.save(self.policy_net.state_dict(), save_dir / filename)
        print(f"Saved model as: {filename}")

    def load_weights(self, suffix=None):
        """Load model weights from a specific version or most recent"""
        save_dir = Path("scripts/our_agent/models")
        
        if suffix:
            filename = f"dqn_{suffix}.pt"
        else:
            # Try to load the most recent version
            model_files = list(save_dir.glob("dqn_*.pt"))
            if not model_files:
                print("No saved model found. Starting with fresh weights.")
                return
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            filename = latest_model.name
        
        try:
            model_path = save_dir / filename
            self.policy_net.load_state_dict(torch.load(model_path, weights_only=True))
            self.target_net.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"Successfully loaded weights from {filename}")
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
            'memory_size': self.memory.memory.maxlen
        }

    def get_model_info(self):
        """Return model architecture information"""
        if self.policy_net is None:
            return {}
        return self.policy_net.get_architecture_info()
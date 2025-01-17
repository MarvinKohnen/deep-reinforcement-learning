import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from collections import deque
import logging
import time
import json
from pathlib import Path

class TrainingLogger:
    def __init__(self, window_size=100, save_dir='training_logs', fresh=False, agent=None, scenario=None):
        # Create base save directory
        self.base_dir = Path(save_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Get or create model timestamp
        if fresh:
            # New model gets new timestamp
            self.model_timestamp = time.strftime("%Y%m%d_%H%M%S")
        else:
            # Try to get timestamp from existing model
            if agent and agent.training_timestamp:
                self.model_timestamp = agent.training_timestamp
            else:
                # Try to find most recent model directory
                model_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('model_')]
                if model_dirs:
                    latest_model = max(model_dirs, key=lambda x: x.name)
                    self.model_timestamp = latest_model.name.replace('model_', '')
                else:
                    self.model_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        self.model_dir = self.base_dir / f"model_{self.model_timestamp}"
        self.model_dir.mkdir(exist_ok=True)
        
        # Create timestamped directory for this run
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = self.model_dir / f"run_{run_timestamp}"
        self.save_dir.mkdir(exist_ok=True)
        
        # Import settings for board size
        from bomberman_rl.envs import settings
        
        # Store agent reference
        self.agent = agent
        
        # Initialize config
        self.config = {
            'model': {},
            'hyperparameters': agent.q_learning.get_hyperparameters() if agent and agent.q_learning else {},
            'reward_mapping': agent.get_reward_mapping() if agent else {},
            'env': {
                'scenario': scenario if scenario else 'classic',
                'board_size': {
                    'rows': settings.ROWS,
                    'cols': settings.COLS
                }
            }
        }
        
        if fresh:
            # Start with fresh stats
            self.rewards = []
            self.losses = []
            self.eps_values = []
            self.steps = []
            self.episode_lengths = []
            self.episode_offset = 0
        else:
            # Try to find most recent run's stats
            prev_runs = sorted([d for d in self.model_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
            if prev_runs:
                latest_run = prev_runs[-1]
                stats_file = latest_run / 'training_stats.json'
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        saved_stats = json.load(f)
                        self.rewards = saved_stats.get('rewards', [])
                        self.losses = saved_stats.get('losses', [])
                        self.eps_values = saved_stats.get('eps_values', [])
                        self.steps = [int(x) for x in saved_stats.get('steps', [])]
                        self.episode_lengths = saved_stats.get('episode_lengths', [])
                        self.episode_offset = int(self.steps[-1] + 1) if self.steps else 0
                else:
                    self.rewards = []
                    self.losses = []
                    self.eps_values = []
                    self.steps = []
                    self.episode_lengths = []
                    self.episode_offset = 0
            else:
                # No previous runs found
                self.rewards = []
                self.losses = []
                self.eps_values = []
                self.steps = []
                self.episode_lengths = []
                self.episode_offset = 0
        
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_losses = deque(maxlen=window_size)

    def log_episode(self, episode, epsilon, loss, reward, episode_length):
        # Update model info if it's now available
        if self.agent and self.agent.q_learning and self.agent.q_learning.policy_net:
            if not self.config['model']:  # Only update if empty
                self.config['model'] = self.agent.q_learning.policy_net.get_architecture_info()
        
        # Adjust episode number to continue from previous training
        actual_episode = episode + self.episode_offset
        
        # Store data
        self.rewards.append(reward)
        self.losses.append(loss if loss else 0)
        self.eps_values.append(epsilon)
        self.steps.append(actual_episode)
        self.episode_lengths.append(episode_length)
        self.recent_rewards.append(reward)
        if loss:
            self.recent_losses.append(loss)

        self.save_stats()
        self.plot_training(save_only=True)

    def save_stats(self):
        """Save training statistics and configuration to file"""
        def convert_to_native_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_native_types(value) for key, value in obj.items()}
            return obj

        stats = {
            'config': convert_to_native_types(self.config),
            'rewards': [round(float(r), 4) for r in self.rewards],
            'losses': [round(float(l), 4) for l in self.losses],
            'eps_values': [round(float(e), 4) for e in self.eps_values],
            'steps': [int(s) for s in self.steps],
            'episode_lengths': [int(l) for l in self.episode_lengths]
        }
        
        with open(self.save_dir / 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)

    def plot_training(self, save_only=False):
        """Plot training progress using all collected data with smoothed curves"""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 12))  # Changed to 4 rows, 1 column
        
        # Calculate rolling averages with dynamic window size
        max_window = 100
        
        def rolling_average(data):
            smoothed = []
            for i in range(len(data)):
                window = min(i + 1, max_window)  # Use all available data up to max_window
                start_idx = max(0, i + 1 - window)
                smoothed.append(np.mean(data[start_idx:i + 1]))
            return np.array(smoothed)
        
        # Calculate smoothed data
        rewards_smooth = rolling_average(self.rewards)
        losses_smooth = rolling_average(self.losses)
        lengths_smooth = rolling_average(self.episode_lengths)
        
        # Plot rewards
        ax1.plot(self.steps, self.rewards, 'black', alpha=0.3, label='Raw')
        ax1.plot(self.steps, rewards_smooth, 'blue', label='Rolling Average')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        
        # Plot losses
        ax2.plot(self.steps, self.losses, 'black', alpha=0.3, label='Raw')
        ax2.plot(self.steps, losses_smooth, 'red', label='Rolling Average')
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        # Plot epsilon (no smoothing)
        ax3.plot(self.steps, self.eps_values)
        ax3.set_title('Epsilon Value')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        
        # Plot episode lengths
        ax4.plot(self.steps, self.episode_lengths, 'black', alpha=0.3, label='Raw')
        ax4.plot(self.steps, lengths_smooth, 'purple', label='Rolling Average')
        ax4.set_title('Episode Lengths')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.save_dir / 'training_progress.png')
        
        if not save_only:
            clear_output(True)
            plt.show()
        plt.close()

    def plot_final_training(self):
        """Plot and save final training curves"""
        self.plot_training(save_only=True) 
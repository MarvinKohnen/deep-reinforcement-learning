import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from collections import deque
import logging
import time
import json
from pathlib import Path
import psutil
import GPUtil
import threading


class TrainingLogger:
    def __init__(self, window_size=100, save_dir='training_logs', fresh=False, agent=None, scenario=None, use_double_dqn=False):
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
                prefix = 'double_' if use_double_dqn else 'model_'
                model_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
                if model_dirs:
                    latest_model = max(model_dirs, key=lambda x: x.name)
                    self.model_timestamp = latest_model.name.replace(prefix, '')
                else:
                    self.model_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create model directory with appropriate prefix
        prefix = 'double_' if use_double_dqn else 'model_'
        self.model_dir = self.base_dir / f"{prefix}{self.model_timestamp}"
        self.model_dir.mkdir(exist_ok=True)
        
        # Create timestamped directory for this run
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = self.model_dir / f"run_{run_timestamp}"
        self.save_dir.mkdir(exist_ok=True)
        
        # Import settings for board size
        from bomberman_rl.envs import settings
        
        # Store agent reference and DQN type
        self.agent = agent
        self.use_double_dqn = use_double_dqn
        
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
            },
            'dqn_type': 'double' if use_double_dqn else 'single'
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
                    self._init_empty_stats()
            else:
                self._init_empty_stats()
        
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_losses = deque(maxlen=window_size)


         # Initialize lists for resource usage
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_memory = []

        # Start resource monitoring
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self.monitor_resources)
        self.monitor_thread.start()


    def monitor_resources(self):
            while not self.stop_monitoring:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.cpu_usage.append(cpu_percent)

                # RAM usage
                ram_percent = psutil.virtual_memory().percent
                self.ram_usage.append(ram_percent)

                # GPU usage (if available)
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Assuming we're using the first GPU
                    self.gpu_usage.append(gpu.load * 100)
                    self.gpu_memory.append(gpu.memoryUsed)
                else:
                    self.gpu_usage.append(0)
                    self.gpu_memory.append(0)

                time.sleep(1)  # Update every second


    def _init_empty_stats(self):
        """Initialize empty statistics"""
        self.rewards = []
        self.losses = []
        self.eps_values = []
        self.steps = []
        self.episode_lengths = []
        self.episode_offset = 0

    def log_episode(self, episode, epsilon, loss, reward, episode_length):
        # Update model info if it's now available
        if self.agent and self.agent.q_learning and not self.config['model']:
            if self.use_double_dqn:
                # For Double DQN
                if hasattr(self.agent.q_learning, 'policy_net_a'):  # Check if it's actually a double DQN model
                    self.config['model'] = {
                        'policy_net_a': self.agent.q_learning.policy_net_a.get_architecture_info(),
                        'policy_net_b': self.agent.q_learning.policy_net_b.get_architecture_info()
                    }
            else:
                # For Single DQN
                if hasattr(self.agent.q_learning, 'policy_net'):  # Check if it's a single DQN model
                    self.config['model'] = self.agent.q_learning.get_model_info()
        
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
            'episode_lengths': [int(l) for l in self.episode_lengths],
            'dqn_type': 'double' if self.use_double_dqn else 'single',
            'cpu_usage': self.cpu_usage,
            'ram_usage': self.ram_usage,
            'gpu_usage': self.gpu_usage,
            'gpu_memory': self.gpu_memory
        }
        
        with open(self.save_dir / 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)

    def plot_training(self, save_only=False):
        """Plot training progress using all collected data with smoothed curves"""
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(20, 12))  # Changed to 5 rows, 1 column
        
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

        # Plot resource usage
        ax5.plot(self.cpu_usage, label='CPU Usage (%)')
        ax5.plot(self.ram_usage, label='RAM Usage (%)')
        if any(self.gpu_usage):
            ax5.plot(self.gpu_usage, label='GPU Usage (%)')
            ax5_twin = ax5.twinx()
            ax5_twin.plot(self.gpu_memory, label='GPU Memory (MB)', color='r')
            ax5_twin.set_ylabel('GPU Memory (MB)')
        ax5.set_title('Resource Usage')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Usage (%)')
        ax5.legend(loc='upper left')
        if any(self.gpu_usage):
            ax5_twin.legend(loc='upper right')
        
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


    def __del__(self):
        # Stop the resource monitoring thread
        self.stop_monitoring = True
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
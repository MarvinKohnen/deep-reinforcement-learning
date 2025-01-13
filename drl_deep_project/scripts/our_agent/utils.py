import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from collections import deque
import logging
import time
import json
from pathlib import Path

class TrainingLogger:
    def __init__(self, window_size=100, log_interval=10, save_dir='training_logs'):
        self.rewards = []
        self.losses = []
        self.eps_values = []
        self.steps = []
        self.episode_lengths = []
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_losses = deque(maxlen=window_size)
        self.start_time = time.time()
        self.log_interval = log_interval
        
        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup logging with file only (no console output)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_episode(self, episode, epsilon, loss, reward, episode_length):
        # Store data
        self.rewards.append(reward)
        self.losses.append(loss if loss else 0)
        self.eps_values.append(epsilon)
        self.steps.append(episode)
        self.episode_lengths.append(episode_length)
        self.recent_rewards.append(reward)
        if loss:
            self.recent_losses.append(loss)

        # Log to file at intervals
        if episode > 0 and episode % self.log_interval == 0:
            avg_reward = np.mean(self.recent_rewards)
            avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0
            elapsed_time = time.time() - self.start_time
            
            self.logger.info(
                f"Episode {episode:4d} | "
                f"Epsilon: {epsilon:.2f} | "
                f"Avg Loss: {avg_loss:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Time: {elapsed_time:.0f}s"
            )
        
            self.save_stats()
            self.plot_training(save_only=True)

    def save_stats(self):
        """Save training statistics to file"""
        stats = {
            'rewards': [round(r, 4) for r in self.rewards],
            'losses': [round(l, 4) for l in self.losses],
            'eps_values': [round(e, 4) for e in self.eps_values],
            'steps': self.steps,
            'episode_lengths': self.episode_lengths
        }
        with open(self.save_dir / 'training_stats.json', 'w') as f:
            json.dump({k: list(map(float, v)) for k, v in stats.items()}, f)

    def plot_training(self, save_only=False):
        """Plot training progress using all collected data"""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot rewards
        ax1.plot(self.steps, self.rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        
        # Plot losses
        ax2.plot(self.steps, self.losses)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        
        # Plot epsilon
        ax3.plot(self.steps, self.eps_values)
        ax3.set_title('Epsilon Value')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        
        # Plot episode lengths
        ax4.plot(self.steps, self.episode_lengths)
        ax4.set_title('Episode Lengths')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        
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
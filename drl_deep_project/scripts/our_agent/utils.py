import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from collections import deque
import logging
import time
import json
from pathlib import Path

class TrainingLogger:
    def __init__(self, window_size=100, save_dir='training_logs', fresh=False):
        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # If fresh training, archive existing stats
        if fresh:
            # Find most recent model file to get its timestamp
            model_dir = Path('scripts/our_agent/models')
            model_files = list(model_dir.glob("dqn_*.pt"))
            
            # Only proceed with archiving if there are model files and training data
            if model_files and any([
                (self.save_dir / 'training_stats.json').exists(),
                (self.save_dir / 'training_progress.png').exists(),
                (self.save_dir / 'training.log').exists()
            ]):
                # Extract timestamp from the most recent model file
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                timestamp = latest_model.name.replace('dqn_', '').replace('.pt', '')
                print(f"\nArchiving previous training run with timestamp {timestamp}")
                
                # Create archive directory with model timestamp
                archive_dir = self.save_dir / f"archive_{timestamp}"
                archive_dir.mkdir(exist_ok=True)
                
                # Move existing files to archive
                if (self.save_dir / 'training_stats.json').exists():
                    (self.save_dir / 'training_stats.json').rename(archive_dir / 'training_stats.json')
                if (self.save_dir / 'training_progress.png').exists():
                    (self.save_dir / 'training_progress.png').rename(archive_dir / 'training_progress.png')
                if (self.save_dir / 'training.log').exists():
                    (self.save_dir / 'training.log').rename(archive_dir / 'training.log')
                    
                print(f"Previous training logs archived to: {archive_dir}")
            else:
                print("\nNo previous training data found")
                
            print("Starting fresh training run with new network!\n")
            
            # Start with fresh stats
            self.rewards = []
            self.losses = []
            self.eps_values = []
            self.steps = []
            self.episode_lengths = []
            self.episode_offset = 0
        else:
            # Try to load existing stats
            stats_file = self.save_dir / 'training_stats.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    saved_stats = json.load(f)
                    self.rewards = saved_stats.get('rewards', [])
                    self.losses = saved_stats.get('losses', [])
                    self.eps_values = saved_stats.get('eps_values', [])
                    self.steps = [int(x) for x in saved_stats.get('steps', [])]
                    self.episode_lengths = saved_stats.get('episode_lengths', [])
                    
                    # If there are existing steps, next episode should continue from last one
                    self.episode_offset = int(self.steps[-1] + 1) if self.steps else 0
            else:
                self.rewards = []
                self.losses = []
                self.eps_values = []
                self.steps = []
                self.episode_lengths = []
                self.episode_offset = 0
        
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_losses = deque(maxlen=window_size)
        self.start_time = time.time()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_episode(self, episode, epsilon, loss, reward, episode_length):
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

        # Log to file at intervals
        if episode > 0 and episode % 10 == 0:
            avg_reward = np.mean(self.recent_rewards)
            avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0
            elapsed_time = time.time() - self.start_time
            
            self.logger.info(
                f"Episode {actual_episode:4d} | "
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
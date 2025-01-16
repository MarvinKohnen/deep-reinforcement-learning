import matplotlib.pyplot as plt
import numpy as np
import json

def parse_log(logfile):
	with open(logfile, 'r') as log:
		stats = json.load(log)
		global losses, rewards, steps, episode_lengths, eps_values
		losses = stats['losses']
		rewards = stats['rewards']
		steps = stats['steps']
		episode_lengths = stats['episode_lengths']
		eps_values = stats['eps_values']

def plot_training():
	"""Plot training progress using all collected data with smoothed curves"""
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 12))  # Changed to 4 rows, 1 column

	# Calculate rolling averages with dynamic window size
	max_window = 1000

	def rolling_average(data):
		smoothed = []
		for i in range(len(data)):
			window = min(i + 1, max_window)  # Use all available data up to max_window
			start_idx = max(0, i + 1 - window)
			smoothed.append(np.mean(data[start_idx:i + 1]))
		return np.array(smoothed)

	# Calculate smoothed data
	rewards_smooth = rolling_average(rewards)
	losses_smooth = rolling_average(losses)
	lengths_smooth = rolling_average(episode_lengths)


#	rewards_trend = [rewards_smooth[100*i] for i in range(0,len(steps)//100)]
#	steps_trend = [steps[100*i] for i in range(0, len(steps)//100)]

	'''
	p = np.polyfit(steps_trend, rewards_trend, 50)
	print(p)
	x_values = np.linspace(0,len(steps), len(steps_trend)*2)
	ax1.plot(x_values, [sum(p[i]*x**i for i in range(len(p))) for x in x_values])
	'''
	#ax1.plot(rolling_average(steps_trend), rewards_trend, 'blue')

	# Plot rewards
	ax1.plot(steps, np.array(rewards)/np.array(episode_lengths), 'black', alpha=0.3, label='Raw')
	ax1.plot(steps, rewards_smooth/lengths_smooth, 'blue', label='Rolling Average')
	ax1.set_title('Episode Rewards per time step')
	ax1.set_xlabel('Episode')
	ax1.set_ylabel('Reward')
#	ax1.set_ylim([-11, -4])
	ax1.legend()

	# Plot losses
	ax2.plot(steps, losses, 'black', alpha=0.3, label='Raw')
	ax2.plot(steps, losses_smooth, 'red', label='Rolling Average')
	ax2.set_title('Training Loss')
	ax2.set_xlabel('Episode')
	ax2.set_ylabel('Loss')
	ax2.legend()

	# Plot epsilon (no smoothing)
	ax3.plot(steps, eps_values)
	ax3.set_title('Epsilon Value')
	ax3.set_xlabel('Episode')
	ax3.set_ylabel('Epsilon')

	# Plot episode lengths
	ax4.plot(steps, episode_lengths, 'black', alpha=0.3, label='Raw')
	ax4.plot(steps, lengths_smooth, 'purple', label='Rolling Average')
	ax4.set_title('Episode Lengths')
	ax4.set_xlabel('Episode')
	ax4.set_ylabel('Steps')
	ax4.legend()

	plt.tight_layout()

	# Save plot
	plt.savefig('results.png')
	#plt.show()
	#plt.close()

parse_log("./training_logs/training_stats.json")
plot_training()
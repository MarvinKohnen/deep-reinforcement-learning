# Bomberman Challenge:

This code was written and submitted by Julius Ferber, Marvin Kohnen and Moritz Gehring

### Contents:

The files are of the following structure:

📁 our_agent/
├── 📄 README.md                             # Project documentation and structure overview
├── 📄 agent.py                              # Main agent implementation with DQN/Double DQN support
├── 📄 double_q_learning.py                  # Double DQN implementation with prioritized experience replay
├── 📄 q_learning.py                         # Single DQN implementation with prioritized experience replay
├── 📄 utils.py                              # Training utilities and visualization tools
└── 📁 models/                               # Directory for saved model weights
    ├── 📄 dqn_[timestamp].pt                # Single DQN model weights
└── 📁 training_logs/                        # Directory for training stats and plots
    ├── 📁 model_[timestamp]                 # Folder for each model   
        ├── 📁 run_[timestamp]               # Folders for multiple training runs with the same model
            ├── 📄 training_progress.png     # Plots for reward, loss, epsilon decay and episode length
            ├── 📄 training_stats.json       # File for architecture, hyperparameters and training stats used for plotting

📄 main.py                                   # Adjusted main for using flags specified in argparsing.py
📄 argparsing.py                             # Added flags for more control over the training 
📄 requirements.txt                          # Requirements for development environment 
    

Key Components:
- The agent supports both single DQN and double DQN architectures
- Uses prioritized experience replay for better sample efficiency
- Includes comprehensive training logging and visualization
- Model weights are saved with timestamps for version control
- Training runs are visualized and training stats are saved
- Added training and evaluating different agents by introducing new flags
- Implements reward shaping and state optimization for the Bomberman environment

### Instructions on how to train, run and test the agent:

Added flags, other than the ones given are:

--weights: Specify 'fresh' for new weights or timestamp (e.g., '20240315_143022') to load specific checkpoint
--use-double-dqn: Use double DQN instead of single DQN 

Otherwise, training goes as specified:

``` python scripts/main.py --train```

plus additional flags as seen above

For running and testing the model, do:

``` python scripts/main.py --weights <timestamp>```


### Development environment

We used the following versions of these most important libraries:

- Python 3.12.8
- PyTorch 2.5.1
- Numpy 2.2.1

For a detailed version of all packages used, take a look at requirements.txt

Disclaimer: Helper code like utils.py was coded with the help of Claude-3.5-Sonnet!

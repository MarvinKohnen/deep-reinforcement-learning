# Q-Function Examples with PyTorch and TensorFlow

The two files are examples that implement Q-learning and use a neural network as a function approximator for the Q-function. The neural network predicts Q-values for all possible actions, given the current state, and is trained to minimize the temporal difference (TD) error. There are two versions - one for pytorch and one in tensorflow. As a recommendation: This was setup in pytorch and afterwards, basically, translated. The pytorch version appears (on a Mac) more performant and a little more stable (both are quite sensitive to hyperparameters - in particular, as we are dealing with learning steps for single examples!). So, if you are in doubt: go with the pytorch version ...

## PyTorch Implementation

The PyTorch implementation uses `torch.optim.Adam` for optimization. It performs well out of the box but shows sensitivity to hyperparameters due to the lack of a replay buffer and the fact that training is performed on single transitions instead of batches. 

## TensorFlow Implementation
In the TensorFlow implementation, I used  `tf.keras.optimizers.SGD` which improved stabilization (compared to `Adam`). This change highlights the sensitivity of the Q-learning setup to hyperparameter tuning. The absence of batch training or a replay buffer makes the training process less robust.

---

## Sensitivity to Hyperparameters

Both implementations are sensitive to:
1. **Learning Rate:** Too high a learning rate can lead to divergence, while too low slows down convergence (and in some cases it didn't pick up learning at all for me).
2. **Optimizer Choice:** The TensorFlow version benefits from SGD, whereas Adam performed better in the PyTorch version.
3. **Single-Example Updates:** Training on single examples, rather than batches, amplifies instability.

---

## Suggestions for Improvement

You can use these programs to build your own DQN approach. Some suggested small improvements:

1. **Replay Buffer:** Introduce a replay buffer to store transitions and sample mini-batches for training. This helps decorrelate samples and stabilizes learning.
2. **Batch Updates:** Train on mini-batches of transitions instead of single samples to make gradient updates smoother and more robust.
3. **Epsilon Decay:** Use a decaying epsilon in the ε-greedy policy to balance exploration and exploitation more effectively over time.

Of course, you can also use a target network and introduce further improvements. You can use the simple example environment to experiment a little bit with small changes in hyperparameters (and will see that this is quite sensitive - e.g. learning rate, batch size (and these two are somehow interconnected), buffer size, epsilon decay ...).

## Feedback

If you find errors or further improvements (on efficiency, particular in tensorflow): Please, post a remark, hint or question to the forum.

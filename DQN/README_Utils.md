## ReplayMemory

The `ReplayMemory` is representative of the experience replay memory used in training the DQN model.

## DQN

The `DQN` class defines a DQN feed forward network with 2 hidden layers and specifiable amounts of neurons in the input, hidden, and output layers.

## DQN3

The `DQN` class defines a DQN feed forward network with 3 hidden layers and specifiable amounts of neurons in the input, hidden, and output layers.

## Hyperparameters

The Hyperparameters used for training the DQN model are stored in the `utils.py` script and are imported when needed. Further, the static starting position the drones where trained on is stored in the file and an arbitrary goal position.

## Usage

Creating a DQN policy and target network:
```python
from environment import World

# Create the training environment
env = World()

# Obtain number of inputs and outputs to DQN model
n_observations = env.reset().process(device='cpu', ['prev_position'])
n_actions = env.action_space.n

# Create the DQN model
policy_net = DQN3(n_observations, n_actions, 512).to(device)
target_net = DQN3(n_observations, n_actions, 512).to(device)
target_net.load_state_dict(policy_net.state_dict())
```
_Note: further usage can be seen in `train.py`_

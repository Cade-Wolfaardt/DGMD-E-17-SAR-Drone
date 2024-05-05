from collections import namedtuple, deque
import random
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a
#           slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Define world parameters:
# goal: The point of interest for the drone
goal = (3779, 4583, 6528)
# Start position of the drone (x,y,z)
start_position = (5000, 5000, 5000)

# Define a transition which maps a (state, action) pair to a
# (next state, reward) pair
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    Replay memory for storing and sampling transitions.
    """
    def __init__(self, capacity: int) -> None:
        """
        Initializes the replay memory.

        Parameters:
        - capacity (int): Maximum capacity of the memory.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """
        Saves a transition in the memory.

        Parameters:
        - *args: Transition tuple or arguments.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        """
        Samples transitions from the memory.

        Paramters:
        - batch_size (int): Size of the batch to sample.

        Returns:
        - list: List of sampled transitions.
        """

        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Returns the current size of the memory.

        Returns:
        - int: Size of the memory.
        """
        return len(self.memory)


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model for reinforcement learning.

    This class defines a DQN model with two hidden layers of variable neuron
    size.

    Methods:
    - forward(x): Forward pass through the network.
    """
    def __init__(
            self, n_observations: int, n_actions: int,
            hidden_nodes: int) -> None:
        """
        Initialize the DQN model with specified layers.

        Paramters:
        - n_observations (int): Number of input observations.
        - n_actions (int): Number of output actions.
        - hidden_nodes (int): Number of nodes in the hidden layers.
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_nodes)
        self.layer2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.layer3 = nn.Linear(hidden_nodes, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Parameters:
        - x (Tensor): Input tensor to the network.

        Returns:
        - Tensor: Output tensor from the network.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQN3(nn.Module):
    """
    Deep Q-Network (DQN) model for reinforcement learning.

    This class defines a DQN model with three hidden layers of variable neuron
    size.

    Methods:
    - forward(x): Forward pass through the network.
    """
    def __init__(
            self, n_observations: int, n_actions: int,
            hidden_nodes: int) -> None:
        """
        Initialize the DQN model with specified layers.

        Paramters:
        - n_observations (int): Number of input observations.
        - n_actions (int): Number of output actions.
        - hidden_nodes (int): Number of nodes in the hidden layers.
        """
        super(DQN3, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_nodes)
        self.layer2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.layer3 = nn.Linear(hidden_nodes, hidden_nodes)
        self.layer4 = nn.Linear(hidden_nodes, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Parameters:
        - x (Tensor): Input tensor to the network.

        Returns:
        - Tensor: Output tensor from the network.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

import sys
import os
import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.optim as optim

from environment import World, LOG_DIR, State
from utils import *

# Check if in Google Colab
IN_COLAB = 'google.colab' in sys.modules
# Mount drive
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive/')

MODEL_DIR = 'Models' if not IN_COLAB else LOG_DIR

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


def select_action(
        state: State, steps_done: int, policy_net: torch.nn.Module,
        env: World, device: str) -> torch.Tensor:
    """
    Select an action based on the current state, exploration strategy, and
    policy network.

    Parameters:
    - state (State): The current state of the environment.
    - steps_done (int): The number of steps done in the exploration.
    - policy_net (torch.nn.Module): The policy network used for action
                                    selection.
    - env (World): The environment in which the drone operates.
    - device (str): The device (CPU or GPU) to use for computations.

    Returns:
    - torch.Tensor: The selected action tensor.
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # The second column on the max result is the index of where
            # the max element was found, so we pick the action with the
            # larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space_sample()]],
            device=device,
            dtype=torch.long
        )


def optimize_model(
        memory: ReplayMemory, policy_net: torch.nn.Module,
        target_net: torch.nn.Module, optimizer: optim.AdamW,
        device: str) -> None:
    """
    Optimize the Q-network model based on a batch of experiences from memory.

    Args:
    - memory (ReplayMemory): The replay memory containing experiences.
    - policy_net (torch.nn.Module): The policy network to optimize.
    - target_net (torch.nn.Module): The target network for computing target
                                    Q-values.
    - optimizer (optim.adamw.AdamW): The optimizer for updating the model's
                                     weights.
    - device (str): The device (CPU or GPU) to use for computations.
    """
    if len(memory) < BATCH_SIZE:
        return
    # Sample a batch of transitions from the memory
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # Compute mask for non-final states and concatenate batch elements
    non_final_mask = torch.tensor(
        tuple(map(lambda state: state is not None, batch.next_state)),
        device=device,
        dtype=torch.bool
    )
    non_final_next_states = torch.cat(
        [state for state in batch.next_state if state is not None]
    )
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q-values for state-action pairs
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute expected state-action values
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(
        state_action_values,
        expected_state_action_values.unsqueeze(1)
    )
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def plot_steps(
        episode_steps: list, episode_success: list, is_ipython: bool,
        show_result: bool = False) -> None:
    """
    Plot the number of steps taken per episode, distinguishing between
    successful and unsuccessful episodes.

    Parameters:
    - episode_steps (list): List of steps taken in each episode.
    - episode_success (list): List indicating success or failure of each
                              episode.
    - is_ipython (bool): Whether the code is running in an IPython environment.
    - show_result (bool, optional): Whether to show the result plot. Default is
                                    False.
    """
    plt.figure(1)
    steps_t = torch.tensor(episode_steps, dtype=torch.float)
    success_mask = torch.tensor(episode_success, dtype=torch.bool)

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.plot(steps_t[success_mask].numpy(), 'go', label='Successful Episodes')
    plt.plot(
        steps_t[~success_mask].numpy(), 'rx', label='Unsuccessful Episodes'
    )
    plt.plot(steps_t.numpy())
    # Show the legend
    plt.legend()
    # Pause a bit so that plots are updated
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def main() -> None:
    """
    Main function to train a DQN model in a given environment.

    This function defines and sets up the training environment, initializes the
    neural networks, optimizer, replay memory, and conducts the training
    process over a number of episodes.

    The training progress is monitored by storing episode steps and success
    metrics, and plotting them to visualize training progress.

    Finally, the trained policy and target networks are saved.
    """
    # Define the device to be used (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate a training environment
    verbosity = 0
    env = World(goal, start_position, verbosity)
    file_id = env.file_id
    # Get number of actions from action space
    n_actions = env.action_space.n
    # Get the number of state observations
    avoid = ['prev_position']
    n_observations = len(env.reset().process(device, avoid))
    # Define policy and target networks for training
    policy_net = DQN3(n_observations, n_actions, 512).to(device)
    target_net = DQN3(n_observations, n_actions, 512).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    # Define optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    # Initialize steps_done for eps_threshold
    steps_done = 0
    # Instantiate a list to contain distances from goal to monitor training
    # progress
    episode_steps = []
    episode_success = []
    # Define the number of episodes depending on available device
    if torch.cuda.is_available():
        num_episodes = 1500
    else:
        num_episodes = 1

    # Begin training
    for i_episode in range(num_episodes):
        # Assign a random new goal
        env.assign_random_goal(distance_val=range(250, 1001))
        # Initialize the environment and get its state
        state = env.reset().process(device, avoid).unsqueeze(0)
        # Count number of steps per episode for monitoring progress
        episode_step_count = 0
        # Execute the episode
        for t in count():
            episode_step_count += 1
            # Select an action
            action = select_action(state, steps_done, policy_net, env, device)
            # Determine consequence of action
            observation, reward, done = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            # If the consequence is terminal then no next state otherwise,
            # define next state as state obtained from carrying out the action
            if done:
                next_state = None
            else:
                next_state = observation.process(device, avoid).unsqueeze(0)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, policy_net, target_net, optimizer, device)
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = (
                    policy_net_state_dict[key]*TAU
                    + target_net_state_dict[key]*(1-TAU)
                )
            target_net.load_state_dict(target_net_state_dict)

            # If end of episode store mission outcome and the number of steps
            # executed in the episode
            if done:
                episode_success.append(env.goal_achieved)
                episode_steps.append(episode_step_count)
                plot_steps(episode_steps, episode_success, is_ipython)
                break
    # End of training sequence
    print('Complete')
    plot_steps(episode_steps, episode_success, is_ipython, show_result=True)
    plt.ioff()
    plt.show()
    # Save the policy network and target network
    policy_net_path = os.path.join(
        MODEL_DIR, f'{file_id}_policy_net.pth'
    )
    target_net_path = os.path.join(
        MODEL_DIR, f'{file_id}_target_net.pth'
    )
    torch.save(policy_net.state_dict(), policy_net_path)
    torch.save(target_net.state_dict(), target_net_path)


if __name__ == '__main__':
    main()

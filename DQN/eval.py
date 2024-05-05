import os
import argparse
import torch
import keyboard
from collections import deque
import time
import pandas as pd
from textwrap import dedent

from environment import World
from utils import DQN, DQN3, goal, start_position

# Known networks and associated architectures: neurons, layers
NETWORKS = {
    'final_DQN': (512, 2),
    '_27_16_10': (512, 3),
    '_26_23_6': (512, 2),
    '_25_21_5': (256, 3),
    '_25_17_48': (512, 2),
    '_24_21_57': (512, 2),
    '_24_19_41': (256, 2),
    '_24_16_52': (128, 2),
    '_23_22_5': (64, 2),
}


def parse_range(range_str: str) -> range:
    """
    Parse a string representing a range into a range object.

    Parameters:
        range_str (str): String representing the range in the format
                         'start-end'.

    Returns:
        range: Range object representing the specified range.

    Raises:
        argparse.ArgumentTypeError: If the input range string is not in the
                                    correct format.
    """
    try:
        # Split the range string into start and end integers
        start, end = map(int, range_str.split('-'))
        # Create a range object from start to end (inclusive)
        return range(start, end + 1)
    except ValueError:
        # Raise an error if the input range string is not in the correct format
        raise argparse.ArgumentTypeError(
            "Range must be in the format 'start-end'"
        )


def parse_bool(bool_str: str) -> bool:
    """
    Parse a string into a boolean value.

    Parameters:
    - bool_str (str): A string representing a boolean value, such as 'true',
                      'false', 't', 'f', etc.

    Returns:
    - bool: The boolean value parsed from the input string.

    Raises:
    - argparse.ArgumentTypeError: If the input string does not represent a
                                  valid boolean value.
    """
    # Convert the input string to lowercase for case-insensitive comparison
    bool_str = bool_str.lower()
    if bool_str in ['true', 't', '1', 'yes', 'y']:
        return True
    elif bool_str in ['false', 'f', '0', 'no', 'n']:
        return False
    else:
        # Raise an error if the input string does not represent a valid boolean
        # value
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_opt() -> argparse.ArgumentParser:
    """
    Parse command-line arguments for the model testing script.

    Returns:
        argparse.ArgumentParser: Argument parser with defined arguments.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument(
        '--id', type=str, default='final_DQN', help="Model ID"
    )
    parser.add_argument(
        '--model_dir', type=str, default='Models',
        help=(
            'Name of the folder containing models in the current working '
            'directory'
        )
    )
    parser.add_argument(
        '--neurons', type=int,
        help='Number of neurons in the model if not known'
    )
    parser.add_argument(
        '--layers', type=int,
        help='Number of hidden layers in the model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=10,
        help='Number of test episodes'
    )
    parser.add_argument(
        '--mode', type=str, default='random',
        help='Modes for setting distance from the goal'
    )
    parser.add_argument(
        '--test_mode', type=str, default='batch',
        help='Whether to run a single test or batch test'
    )
    parser.add_argument(
        '--multiplier', type=float, default=1,
        help='Amount to increase distance each episode if mode = ascending'
    )
    parser.add_argument(
        '--verbosity', type=int, default=0,
        help='Whether to log information on the testing environment'
    )
    parser.add_argument(
        '--render', type=parse_bool, default=False,
        help='Whether to render on single test completion'
    )
    parser.add_argument(
        '--goal_dist', type=int, default=500,
        help=(
            'Distance from start to goal in single and constant batch modes'
        )
    )
    parser.add_argument(
        '--goal_range', type=parse_range, default=range(250, 1001),
        help='Range for batch test distances'
    )
    parser.add_argument(
        '--step_length', type=int, default=10,
        help='Step size taken each action in the environment, units arbitrary'
    )
    parser.add_argument(
        '--threshold', type=float, default=40,
        help='Threshold distance for achieving the mission'
    )

    # Parse and return the parsed arguments
    return parser.parse_args()


def input_handling(opt: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Handle input parameters for the model testing script.

    This function checks if the model ID is known or not, and adjusts the
    neurons and layers accordingly.

    Parameters:
        opt (argparse.ArgumentParser): Parsed command-line arguments.

    Returns:
        argparse.ArgumentParser: Updated argument parser object.
    """
    # Check if the model ID is not known and required parameters are missing
    if opt.id not in NETWORKS and (opt.layers is None or opt.neurons is None):
        print("Model not known, please provide --layers and --neurons")
        exit(0)  # Exit the program if required parameters are missing

    # If the model ID is known, update neurons and layers based on the
    # predefined values
    elif opt.id in NETWORKS:
        opt.neurons = NETWORKS[opt.id][0]
        opt.layers = NETWORKS[opt.id][1]

    # If rendering is set to True and not carrying out a batch test keep logs
    # to make rendering possible
    if opt.render and opt.test_mode == 'single':
        opt.verbosity = 2

    # Return the updated argument parser object
    return opt


class Eval(object):
    """
    Class for evaluating a model using various testing methods.

    This class provides functionality to evaluate a model using different
    testing methods, including single tests and batch tests with configurable
    parameters.
    """
    def __init__(
            self, id: str = '_27_16_10', model_dir: str = 'Models',
            neurons: int = 512, layers: int = 2, batch_size: int = 10,
            mode: str = 'random', test_mode: str = 'batch',
            multiplier: float = 1, verbosity: int = 0, render: bool = False,
            goal_dist: int = 500, goal_range: range = range(250, 1001),
            step_length: int = 10, threshold: int = 40) -> None:
        """
        Initialize the testing environment for the drone.

        Parameters:
        - id (str): Model ID.
        - model_dir (str): Directory containing models.
        - neurons (int): Number of neurons in the model.
        - layers (int): Number of hidden layers in the model.
        - batch_size (int): Number of test episodes.
        - mode (str): Modes for setting distance from goal.
        - test_mode (str): Whether to run a single test or batch test.
        - multiplier (float): Amount to increase distance each episode if mode
                              is ascending.
        - verbosity (int): Verbosity level for logging information.
        - render (bool): Whether to render on single test completion.
        - goal_dist (int): Distance from start to goal in single and constant
                           batch modes.
        - goal_range (range): Range for batch test distances.
        - step_length (int): Step size taken each action in environment.
        - threshold (int): Threshold distance for achieving mission.
        """
        # Assign parameters to instance variables
        self.id = id
        self.model_dir = model_dir
        self.test_mode = test_mode
        self.batch_size = batch_size
        self.mode = mode
        self.multiplier = multiplier
        self.render = render
        self.goal_distance = goal_dist
        self.goal_range = goal_range

        # Determine the model class based on the number of layers
        Model = DQN3 if layers == 3 else DQN

        # Initialize environment and policy network
        self.env = World(goal, start_position, verbosity)
        self.env.step_length = step_length
        self.env.threshold = threshold
        self.n_actions = self.env.action_space.n
        self.state = self.env.reset()
        self.avoid = ['prev_position']
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_observations = len(self.state.process(self.device, self.avoid))
        self.policy_net = Model(
            self.n_observations,
            self.n_actions,
            neurons,
        ).to(self.device)

        # Load pre-trained policy network
        self.policy_net.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, f'{self.id}_policy_net.pth'),
                map_location=torch.device(self.device)
            )
        )
        self.policy_net.eval()

    def plan_path(self) -> tuple:
        """
        Plan the drone's path using the DQN policy.

        This method executes actions based on the DQN's policy until a
        termination condition is met.

        Returns:
        - tuple: A tuple containing goal achieved status and termination
                 status.
        """
        # Initialize termination condition and counters
        terminate = self.env.collision or self.env.goal_achieved
        i = 0
        position_memory = deque([], maxlen=2)
        repeat_count = 0
        # Loop until termination condition is met
        while not terminate:
            i += 1
            if keyboard.is_pressed('esc'):
                print("ESC key pressed")
                break

            # Process current state and get action from policy network
            self.state = self.state.process(self.device, self.avoid)
            with torch.no_grad():
                action = self.policy_net(self.state).argmax().item()
            # Perform the action in the environment
            self.state, _, done = self.env.step(action)
            terminate = self.env.collision or self.env.goal_achieved
            # Track positions and check for exit conditions
            new_position_pair = (
                tuple(self.state['prev_position']),
                tuple(self.state['position'])
            )
            if len(position_memory) == 2:
                mem0 = position_memory[0]
                mem1 = position_memory[1][::-1]
                if mem0 == mem1:
                    repeat_count += 1
            if repeat_count > 2 or i > 3000:
                break
            position_memory.append(new_position_pair)

        return self.env.goal_achieved, done

    def batch_test(self) -> None:
        """
        Run a batch of tests and print summary metrics.

        This method runs a batch of tests with random or specified goal
        distances, collects results, calculates metrics, and prints out summary
        information.
        """
        # Start timing the batch test
        t0 = time.time()
        # Determine the goal distance based on the mode
        goal_distance = self.goal_distance
        if self.mode == 'random':
            goal_distance = self.goal_range
        # Run the batch tests and collect results
        results, distances, initial_distances = (
            self.run_batch_tests(goal_distance)
        )
        # Stop timing and print elapsed time
        t1 = time.time()
        print(f'Time elapsed:\t{t1 - t0}')
        # Print summary metrics
        self.print_summary(results)
        # Print general metrics if batch size is small
        self.print_general_metrics(results, distances, initial_distances)
        # Print fail metrics
        self.print_fail_metrics(results, initial_distances, distances)

    def run_batch_tests(self, goal_distance: int | range) -> tuple:
        """
        Run a batch of tests with specified or random goal distances.

        This method runs a batch of tests with either a specified goal distance
        or a randomly generated goal distance within a range. It collects and
        returns the results, initial distances, and final distances for each
        test.

        Parameters:
        - goal_distance (int or range): The goal distance to be used in tests.

        Returns:
        - tuple: A tuple containing results, distances, and initial_distances.
        """
        # Initialize lists to store results and distances
        results = []
        distances = []
        initial_distances = []
        # Run the batch of tests
        for test in range(self.batch_size):
            # Check if the escape key is pressed to abort the test
            if keyboard.is_pressed('esc'):
                print("ESC key pressed")
                break
            # Adjust the goal distance if needed
            if not isinstance(goal_distance, range):
                goal_distance = self.multiplier * goal_distance
            # Assign a random goal distance to the environment
            self.env.assign_random_goal(goal_distance)
            # Record the initial distance to the goal
            initial_distances.append(self.env.state['distance_to_goal'])
            # Reset the environment and plan the path
            self.state = self.env.reset()
            results.append(self.plan_path())
            distances.append(self.env.state['distance_to_goal'])

        return results, distances, initial_distances

    def print_summary(self, results: list) -> None:
        """
        Print a summary of test results.

        This method calculates and prints the number of achieved goals, not
        achieved goals, and the accuracy of the test results.

        Parameters:
        - results (list): List of test results, where each result is a tuple
                          (goal_achieved, termination_status).
        """
        # Calculate metrics based on test results
        total = len(results)
        achieved = sum(val[0] for val in results)
        not_achieved = sum(not val[0] for val in results)
        accuracy = achieved / total

        # Generate the summary string
        summary = f"""
        Goal Achieved Count:\t\t\t{achieved}
        Goal Not Achieved Count:\t\t{not_achieved}
        Accuracy:\t\t\t\t{accuracy * 100:.2f}%
        """

        # Format the summary string with dashes and print it
        summary = '-' * 45 + dedent(summary) + '-' * 45
        print(summary)

    def print_general_metrics(
            self, results: list, distances: list,
            initial_distances: list) -> None:
        """
        Print general metrics for individual tests in a batch.

        This method prints metrics for each test in a batch, including episode
        number, distance from goal, termination event, and initial distance.

        Parameters:
        - results (list): List of test results, where each result is a tuple
                          (goal_achieved, termination_status).
        - distances (list): List of distances from goal for each test.
        - initial_distances (list): List of initial distances to the goal for
                                    each test.
        """
        # Check if the batch size is small enough to print individual metrics
        if self.batch_size <= 20:
            # Define the format for general metrics
            general_metrics = (
                'Episode {}. Distance from goal {}. Termination '
                'event {}. Initial distance {}.'
            )
            # Print metrics for each test
            for ind, dist in enumerate(distances):
                print(
                    general_metrics.format(
                        ind, dist, results[ind][1], initial_distances[ind]
                    )
                )

    def print_fail_metrics(
            self, results: list, distances: list,
            initial_distances: list) -> None:
        """
        Print failure metrics for failed test missions.

        This method calculates and prints the average initial distance and
        final distance for failed missions in the batch of tests.

        Parameters:
        - results (list): List of test results, where each result is a tuple
                          (goal_achieved, termination_status).
        - distances (list): List of distances from goal for each test.
        - initial_distances (list): List of initial distances to the goal for
                                    each test.
        """
        # Create a DataFrame to analyze test results
        df = pd.DataFrame({
            'termination': [pair[0] for pair in results],
            'initial_distance': initial_distances,
            'final_distance': distances
        })
        # Filter failed missions
        filter = df.termination == False
        # Calculate mean initial and final distances for failed missions
        mean_initial_distance = df[filter].initial_distance.mean()
        mean_final_distance = df[filter].final_distance.mean()
        # Generate and print failure metrics
        fail_metrics = (
            'Failed missions: average initial distance {}, '
            'final distance {}'
        )
        print(
            fail_metrics.format(mean_initial_distance, mean_final_distance)
        )

    def test(self) -> None:
        """
        Run a single test with a random or specified goal distance.

        This method runs a single test with either a random goal distance or a
        specified goal distance. It assigns the goal distance to the
        environment, resets the state, plans the path, and optionally renders
        the path if rendering is enabled.
        """
        # Assign a random or specified goal distance to the environment
        self.env.assign_random_goal(self.goal_distance)
        # Reset the environment and plan the path
        self.state = self.env.reset()
        _ = self.plan_path()
        # Render the path if rendering is enabled
        if self.render:
            self.env.render_path()

    def run(self) -> None:
        """
        Run the test or batch test based on the mode.

        This method checks the test mode and either runs a single test or a
        batch test based on the mode specified.
        """
        # Check the test mode and run the appropriate test
        if self.test_mode == 'single':
            # Run a single test
            self.test()
        else:
            # Run a batch test
            self.batch_test()


def main(opt: argparse.ArgumentParser) -> None:
    """
    Main function to initialize and run the evaluation process.

    This function creates an Eval object with the provided command-line
    arguments and runs the evaluation process.

    Parameters:
    - opt (argparse.ArgumentParser): Command-line arguments parsed by argparse.
    """
    # Create an Eval object with the provided options
    eval = Eval(
        opt.id, opt.model_dir, opt.neurons, opt.layers, opt.batch_size,
        opt.mode, opt.test_mode, opt.multiplier, opt.verbosity, opt.render,
        opt.goal_dist, opt.goal_range, opt.step_length, opt.threshold
    )

    # Run the evaluation process
    eval.run()


if __name__ == '__main__':
    opt = parse_opt()
    opt = input_handling(opt)
    main(opt)

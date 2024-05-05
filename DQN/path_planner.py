import argparse
import numpy as np
import torch
from pyproj import Geod
import pandas as pd
import re
import os
import copy

from utils import DQN, DQN3
from environment import World, State

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


def parse_tuple(tuple_str: str) -> tuple:
    """
    Parse a tuple string into its consituents.

    Parameters:
    - tuple_str (str): A string representing a tuple of floats in the format
                       'a_b_c'.

    Returns:
    - tuple: A tuple containing floating point values extracted from the tuple
             string.

    Raises:
    - argparse.ArgumentTypeError: If the tuple string is not in the correct
                                  format.
    """
    try:
        # Split the geo-coordinate string and convert each part to float
        a, b, c = map(float, tuple_str.split('_'))
        return (a, b, c)
    except ValueError:
        # Raise an error if the geo-coordinate string is not in the correct
        # format
        raise argparse.ArgumentTypeError(
            "tuple must be in the format 'a_b_c'"
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
    Parse command-line options using argparse and return the parsed arguments.

    Returns:
    - argparse.ArgumentParser: An argparse parser containing the parsed
                               arguments.
    """
    parser = argparse.ArgumentParser()
    # Add command-line arguments
    parser.add_argument(
        '--id', type=str, default='final_DQN', help="Model ID"
    )
    parser.add_argument(
        '--model_dir', type=str, default='Models',
        help='Name of directory containing models'
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
        '--verbosity', type=int, default=0,
        help='Whether to log information on the testing environment'
    )
    parser.add_argument(
        '--render', type=parse_bool, default=False,
        help='Whether to render on single test completion'
    )
    parser.add_argument(
        '--to_txt', type=parse_bool, default=False,
        help='Whether to output a text file of the planned path'
    )
    parser.add_argument(
        '--step_length', type=int, default=10,
        help='Step size for each action in the environment, units arbitrary'
    )
    parser.add_argument(
        '--threshold', type=float, default=40,
        help='Threshold distance for achieving the mission'
    )
    parser.add_argument(
        '--delta_dist', type=float, default=0.5,
        help='Step size taken for each action in the real world, meters'
    )
    parser.add_argument(
        '--start', type=parse_tuple, default=(42.355118, -71.071305, 6),
        help='Geographic coordinates of the start position'
    )
    parser.add_argument(
        '--goal', type=parse_tuple, default=(42.355280, -71.070911, 26.5),
        help='Geographic coordinates of the goal position'
    )
    parser.add_argument(
        '--mission_dir', type=str, default='Missions',
        help='Directory to output the mission path'
    )
    parser.add_argument(
        '--to_csv', type=parse_bool, default=False,
        help='Whether to output a CSV file containing the mission path'
    )
    parser.add_argument(
        '--cartesian_start', type=parse_tuple, default=(5000, 5000, 5000),
        help='The starting position of the drone in the simulation environment'
    )

    return parser.parse_args()


def input_handling(opt: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Handle input options related to neural network parameters.

    Parameters:
    - opt (argparse.ArgumentParser): Parsed command-line arguments.

    Returns:
    - argparse.ArgumentParser: Updated argparse parser with neural network
                               parameters set if necessary.
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
    if opt.render:
        opt.verbosity = 2

    # Return the updated argument parser object
    return opt


class PathPlanner(object):
    """
    Represents a Path Planner for a drone.

    This class serves as an interface to interact with trained DQN models and
    output a path plan.
    """
    def __init__(
            self, mission_dir: str = 'Missions',
            model_dir: str = 'Models', id: str = 'final_DQN',
            geo_start: tuple = (42.355118, -71.071305, 6),
            geo_goal: tuple = (42.355280, -71.070911, 26.5),
            verbosity: int = 0, step_length: int = 10, threshold: int = 40,
            layers: int = 2, neurons: int = 512, delta_dist: float = 0.5,
            cartesian_start: tuple = (5000, 5000, 5000), render: bool = False,
            to_txt: bool = False, to_csv: bool = False):
        """
        Initialize the class with default or provided parameters.

        Paremeters:
        - mission_dir (str): The directory to store mission-related data,
                             default is 'Missions'.
        - model_dir (str): The path to the models directory, default is
                           'Models'.
        - id (str): The ID of the model, default is 'final_DQN'.
        - geo_start (tuple): Geographic coordinates of the start position,
                             default is (42.355118, -71.071305, 6).
        - geo_goal (tuple): Geographic coordinates of the goal position,
                            default is (42.355280, -71.070911, 26.5).
        - verbosity (int): Verbosity level, default is 2.
        - step_length (int): Step size taken for each action in the
                             environment, default is 10.
        - threshold (int): Threshold distance for achieving the mission,
                           default is 40.
        - layers (int): Number of hidden layers in the neural network,
                        default is 2.
        - neurons (int): Number of neurons in the neural network,
                         default is 512.
        - delta_dist (float): Step size taken for each action in the real
                              world, default is 0.5.
        - cartesian_start (tuple): Cartesian coordinate of starting position,
                                   default (5000, 5000, 5000)
        - render (bool): Whether to render the path in simulation environment.
        - to_txt (bool): Whether to output the mission to a txt file.
        - to_csv (bool): Whether to output the mission geodetic points
                         (latitude, longitude) to a csv for plotting on a map.
        """
        # Assign parameters to instance variables
        self.dir = mission_dir
        self.id = id
        self.geo_start = np.array(geo_start)
        self.geo_position = self.geo_start.copy()
        self.geo_goal = np.array(geo_goal)
        self.step_length = step_length
        self.delta_dist = delta_dist
        self.cartesian_start = cartesian_start
        self.render = render
        self.to_txt = to_txt
        self.to_csv = to_csv

        # Determine the model class based on the number of layers
        Model = DQN3 if layers == 3 else DQN
        # Setup mission file if needed
        if to_txt or to_csv:
            self._setup_mission_file()

        # Initialize environment and policy network
        self.geod = Geod(ellps='WGS84')
        self.goal = self._get_cartesian_goal()
        self.env = World(self.goal, cartesian_start, verbosity)
        self.env.step_length = step_length
        self.env.threshold = threshold
        self.n_actions = self.env.action_space.n
        self.state = self.env.reset()
        self.avoid = ['prev_position']
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.n_obs = len(self.state.process(self.device, self.avoid))
        self.policy_net = Model(
            self.n_obs,
            self.n_actions,
            neurons
        ).to(self.device)
        # Load pre-trained policy network
        self.policy_net.load_state_dict(
            torch.load(
                os.path.join(model_dir, f'{id}_policy_net.pth'),
                map_location=torch.device(self.device)
            )
        )
        self.policy_net.eval()

    def _geo_action(self, action: int, geo_position: np.ndarray) -> np.ndarray:
        """
        Perform a geographical action based on the given action code.

        Parameters:
        - action (int): The action code representing the desired movement.
        - geo_position (np.ndarray): The current geographical position as a
                                     NumPy array with latitude, longitude,
                                     and altitude.

        Returns:
        - np.ndarray: The updated geographical position after applying the
                      action.
        """
        # Interpret the action into delta azimuth and move flags
        delta_azimuth, _, geo_move = self.interpret_action(action)
        # Extract latitude and longitude from the geographical position
        lat, lon = geo_position[:-1]
        # If the move flag is True, perform the geographical move
        if geo_move:
            # Calculate the new latitude and longitude based on the delta
            # azimuth and delta distance
            new_lon, new_lat, _ = self.geod.fwd(
                lon, lat, delta_azimuth,
                self.delta_dist, radians=False
            )
            geo_position = np.array((new_lat, new_lon))

        return geo_position

    def _interpret_cart_action(self, action: int) -> np.ndarray:
        """
        Interpret a Cartesian action code into a quadcopter offset vector.

        Parameters:
        - action (int): The action code representing the desired movement
                        direction.

        Returns:
        - np.ndarray: The quadcopter offset vector representing the movement.
        """
        quad_offset = (0, 0)
        # Determine the quadcopter offset based on the action code
        if action == 0:
            # Move right
            quad_offset = (self.step_length, 0)
        elif action == 1:
            # Move forward
            quad_offset = (0, self.step_length)
        elif action == 3:
            # Move left
            quad_offset = (-self.step_length, 0)
        elif action == 4:
            # Move backward
            quad_offset = (0, -self.step_length)

        return np.array(quad_offset)

    def _get_cartesian_goal(self) -> np.ndarray:
        """
        Calculate the Cartesian goal position based on geographical start
        and goal positions.

        Returns:
        - np.ndarray: The Cartesian goal position as a NumPy array with x, y,
                      and z coordinates.
        """
        # Calculate the differences in x and y coordinates between start and
        # goal positions
        dx, dy = self.geo_goal[:-1] - self.geo_start[:-1]
        # Calculate the step size in x and y directions based on geographical
        # actions
        x_step = tuple(
            self._geo_action(0, self.geo_start) - self.geo_start[:-1]
        )[0]
        y_step = tuple(
            self._geo_action(1, self.geo_start) - self.geo_start[:-1]
        )[1]
        # Calculate the number of steps needed in x and y directions
        x_steps = round(dx / x_step)
        y_steps = round(dy / y_step)
        # Generate a list of actions to reach the goal in x and y directions
        actions = []
        if x_steps >= 0:
            actions += [0] * x_steps
        elif x_steps < 0:
            actions += [3] * abs(x_steps)

        if y_steps >= 0:
            actions += [1] * y_steps
        elif y_steps < 0:
            actions += [4] * abs(y_steps)
        # Calculate the Cartesian position based on the generated actions
        cart_position = np.array(self.cartesian_start[:-1])
        for action in actions:
            cart_position += np.array(self._interpret_cart_action(action))
        # Calculate the altitude difference between start and goal positions
        delta_geo_alt = self.geo_goal[-1] - self.geo_start[-1]
        cart_alt_steps = round(delta_geo_alt / self.delta_dist)
        delta_cart_alt = cart_alt_steps * self.step_length
        cart_alt = self.cartesian_start[-1] + delta_cart_alt

        return np.array((*cart_position, cart_alt))

    def _setup_mission_file(self) -> None:
        """
        Set up the mission file for the current session.

        This method checks for existing mission files with the same ID and
        renames them accordingly to avoid overwriting. It generates a new
        mission file name based on the ID and the current count of mission
        files for that ID.
        """
        # Check for existing mission files with the same ID and rename
        # accordingly
        dir = self.dir
        gen_pattern = fr'[^/]*{self.id}[^/]*'
        mission_files = os.listdir(dir)
        matches = [
            name for file in mission_files
            for name in re.findall(gen_pattern, file)
        ]
        if matches:
            file_gen = (
                max(int(file_name.split('_')[-3]) for file_name in matches)
                + 1
            )
        else:
            file_gen = 0
        file_gen = str(file_gen)
        # Generate the mission file name
        self.mission_file = os.path.join(
            dir, f'{self.id}_{file_gen}_mission_file.txt'
        )

    def set_start(self, lat: float, lon: float, alt: float) -> None:
        """
        Set the geographical start position for the environment.

        Parameters:
        - lat (float): Latitude of the start position.
        - lon (float): Longitude of the start position.
        - alt (float): Altitude of the start position.
        """
        # Set the geographical start position and copy it for the current
        # position
        self.geo_start = np.array((lat, lon, alt))
        self.geo_position = self.geo_start.copy()

    def set_goal(self, lat: float, lon: float, alt: float) -> None:
        """
        Set the geographical goal position for the environment and update the
        Cartesian goal.

        Parameters:
        - lat (float): Latitude of the goal position.
        - lon (float): Longitude of the goal position.
        - alt (float): Altitude of the goal position.
        """
        # Set the geographical goal position
        self.geo_goal = np.array((lat, lon, alt))
        # Calculate the Cartesian goal position based on the geographical goal
        self.goal = self._get_cartesian_goal()
        # Update the environment's goal position
        self.env.update_goal(self.goal)

    def setup_mission(self) -> None:
        """
        Setup the mission parameters and generate mission files.

        This method sets up the mission parameters and generates mission files
        based on the mode ('text' or 'list').

        Raises:
        - ValueError: If the mode is not recognized.
        """
        # Extract latitude, longitude, and altitude from the geographical start
        # position
        lat, lon, alt = self.geo_start
        # Mission file start
        lines = [
            'QGC WPL 110',
            f'0\t0\t0\t22\t0\t0\t0\t0\t{lat}\t{lon}\t{alt}\t1\n',
        ]
        self.header = '\n'.join(lines)
        # Mission parameters
        self.mission = {
            'index': 0, 'waypoint': 0, 'frame': 3,
            'command': 82, 'col5': 0, 'col6': 0,
            'col7': 0, 'col8': 0, 'latitude': lat,
            'longitude': lon, 'altitude': alt, 'continue': 1
        }
        self.mission_txt = '{}\t'*8 + '{:.6f}\t'*2 + '{:.1f}\t' + '{}'

        # Store the first mission line in a list
        self.path = copy.copy(self.header)

    def get_action(self, state: State) -> int:
        """
        Get the action to take based on the current state.

        The state is processed to exclude certain features specified in
        self.avoid.

        Parameters:
        - state (State): The current state of the environment.

        Returns:
        - int: The action to take.
        """
        # Process the state to exclude features specified in self.avoid
        state = state.process(self.device, self.avoid)
        # Disable gradient tracking for inference
        with torch.no_grad():
            # Get the action with the highest predicted Q-value
            action = self.policy_net(state).argmax().item()

        return action

    def interpret_action(self, action: int) -> tuple:
        """
        Interpret the action to determine movement offsets.

        Actions are mapped to movement offsets and flags based on predefined
        mappings.

        Parameters:
        - action (int): The action to interpret.

        Returns:
        - tuple: A tuple containing the movement offsets and a flag indicating
                 whether the action involves geographic movement.
        """
        # Initialize the movement offset and flag for geographic movement
        delta_azimuth = 0
        delta_alt = 0
        geo_move = False
        # Map actions to movement offsets and flags
        if action == 0:
            geo_move = True
        elif action == 1:
            delta_azimuth = 90
            geo_move = True
        elif action == 2:
            delta_alt = self.delta_dist
        elif action == 3:
            delta_azimuth = 180
            geo_move = True
        elif action == 4:
            delta_azimuth = 270
            geo_move = True
        elif action == 5:
            delta_alt = -self.delta_dist

        return delta_azimuth, delta_alt, geo_move

    def do_action(self, action: int) -> None:
        """
        Perform the specified action in the environment.

        Actions can involve moving in the geographic space or changing
        altitude, based on the interpretation of the action.

        Parameters:
        - action (int): The action to perform.
        """
        # Interpret the action to determine the movement offsets and flags
        delta_azimuth, delta_alt, geo_move = self.interpret_action(action)
        # Extract the current geographic position components
        lat, lon, alt = self.geo_position
        # Perform the action based on its type
        if geo_move:  # Geographic movement action
            # Calculate the new geographic position using geodetic forward
            # projection
            new_lon, new_lat, _ = self.geod.fwd(
                lon, lat, delta_azimuth,
                self.delta_dist, radians=False
            )
            # Update the geographic position with the new latitude and
            # longitude
            self.geo_position[:-1] = np.array((new_lat, new_lon))
        elif delta_alt != 0:  # Altitude change action
            # Update the altitude by adding the delta altitude
            self.geo_position[-1] += delta_alt

    def get_obs(self) -> None:
        """
        Update mission details and write them to the mission file.

        This method updates the mission index and geographic coordinates,
        and writes the updated mission information to a file.
        """
        # Increment the mission index
        self.mission['index'] += 1
        # Extract the current geographic position components
        lat, lon, alt = self.geo_position
        # Update the mission with the current geographic coordinates
        self.mission['latitude'] = lat
        self.mission['longitude'] = lon
        self.mission['altitude'] = alt
        # Generate the line to write based on the mission information
        line_to_write = self.mission_txt.format(*self.mission.values())
        # Update the mission path
        self.path += line_to_write.strip() + '\n'

    def step(self, action: int) -> None:
        """
        Perform a step in the mission by executing an action and updating
        observations.

        This method performs the specified action, updates the observations,
        and progresses the mission accordingly.

        Parameters:
        - action (int): The action to be executed.
        """
        # Execute the specified action
        self.do_action(action)
        # Update observations based on the action
        self.get_obs()

    def end_sequence(self) -> None:
        """
        End the mission sequence by finalizing the mission and updating
        observations.

        This method sets the final parameters for the mission sequence, updates
        observations, and concludes the mission.
        """
        # Set final mission parameters
        lat, lon = self.geo_start[:-1]
        self.mission['index'] += 1
        self.mission['frame'] = 0
        self.mission['command'] = 20
        self.mission['latitude'] = lat
        self.mission['longitude'] = lon
        self.mission['altitude'] = 0
        # Generate the line to write to the mission file
        line_to_write = self.mission_txt.format(*self.mission.values())
        # Append to path
        self.path += line_to_write.strip()

    def write_to_txt(self) -> None:
        """
        Write the mission path to a text file.

        This method writes the mission path to a text file specified by the
        mission_file attribute.
        """
        # Define the contents to write
        with open(self.mission_file, 'w') as f:
            f.write(self.path)

    def write_to_csv(self) -> None:
        """
        Write the latitude and longitude positions to a CSV file.

        This method extracts relevant columns from the mission path, and
        converts it to CSV format for easier analysis and visualization.
        """
        # Assign the path to a dataframe
        data = [row.split('\t') for row in self.path.split('\n')[1:]]
        df = pd.DataFrame(data)
        # Select specific columns
        cols = [0, 8, 9]
        df = df.iloc[:, cols]
        # Rename columns
        df.columns = ['Index', 'Latitude', 'Longitude']
        # Generate the CSV file name
        csv_file = self.mission_file.replace('txt', 'csv')
        # Save DataFrame to CSV without index
        df.to_csv(csv_file, index=False)

    def run(self) -> list:
        """
        Run the mission sequence.

        This method sets up the mission parameters, writes them to the mission
        file, initializes the mission loop, executes actions based on the
        policy network's decisions, and ends the mission sequence. It
        optionally saves the mission path to a text or CSV file and renders the
        mission path if specified.

        Returns:
        - list: The mission path as a list.
        """
        # Setup the mission parameters and write to the mission file
        self.setup_mission()
        # Get the initial state
        state = self.state
        # Initialize the done flag for the mission loop
        done = False
        # Main mission loop
        while not done:
            # Get action from the policy network based on the current state
            action = self.get_action(state)
            # Execute the action in the environment and get the next state
            state, _, done = self.env.step(action)
            # Perform the action in the mission
            self.step(action)
        # End the mission sequence
        self.end_sequence()
        # Optionally save the mission path to a txt file
        if self.to_txt:
            self.write_to_txt()
        # Optionally save the mission path to a CSV file
        if self.to_csv:
            self.write_to_csv()
        # Optionally render the mission path
        if self.render:
            self.env.render_path()

        return self.path


def main(opt: argparse.ArgumentParser) -> None:
    """
    Main function to run the mission based on the provided options.

    This function initializes the path planner and runs the mission generating
    a mission path.

    Parameters:
    - opt (argparse.ArgumentParser): Parsed command-line arguments.
    """
    # Initialize the model with provided options
    path_planner = PathPlanner(
        opt.mission_dir, opt.model_dir, opt.id, opt.start, opt.goal,
        opt.verbosity, opt.step_length, opt.threshold, opt.layers,
        opt.neurons, opt.delta_dist, opt.cartesian_start, opt.render,
        opt.to_txt, opt.to_csv
    )
    # Obtain and output the path
    path = path_planner.run()


if __name__ == '__main__':
    opt = parse_opt()
    opt = input_handling(opt)
    main(opt)

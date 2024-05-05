import numpy as np
import os
import sys
import itertools
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
from textwrap import dedent
from collections import deque
import pandas as pd
from torch import Tensor, tensor, float32, cat

# Assign directory to save logs depending platform
LOG_DIR = (
    '/content/drive/My Drive/' if 'google.colab' in sys.modules else 'Logs'
)


class Cylinder(object):
    """A class representing a cylinder object."""

    def __init__(self, radius: float, center: tuple, height: float) -> None:
        """
        Initialize the Cylinder with radius, center coordinates, and height.

        Parameters:
        - radius (float): The radius of the cylinder
        - center (tuple): The cartesian coordinates of the center of the
                          cylinder's base, (x, y, z)
        - height (float): The height of the cylinder
        """
        self.radius = radius
        self.center = center
        self.height = height


class State(dict):
    """
    Represents a state dictionary with additional processing capabilities.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the State object.

        Parameters:
        - *args, **kwargs: Any arguments and keyword arguments accepted by the
                           dictionary constructor.
        """
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        """
        Returns a string representation of the State object.

        Returns:
        - str: String representation.
        """
        return f"State({super().__repr__()})"

    def process(
            self, device: str, rem_keys: list | None = None) -> Tensor:
        """
        Processes the state dictionary and returns a Tensor.

        Parameters:
        - device (str): Device to use for the tensor.
        - rem_keys (list, optional): List of keys to remove from the state.
                                     Defaults to None.

        Returns:
        - Tensor: Processed state tensor.
        """
        # Create a copy of the state dictionary
        state = State(self.copy())
        # Remove specified keys from the state
        if rem_keys is not None:
            for val in rem_keys:
                if val in state:
                    del state[val]
        # Convert non-iterable values to lists for later concatenation
        for key in state:
            if isinstance(state[key], (int, float)):
                state[key] = [state[key]]
            elif isinstance(state[key], (bool, np.bool_)):
                state[key] = [1] if state[key] else [0]
        # Concatenate values into a tensor
        state = cat(
            [tensor(state[key], dtype=float32, device=device) for key in state]
        )

        return state


class Discrete(set):
    """
    A subclass of set representing a discrete set of elements.

    Attributes:
    - n (int): The number of elements in the discrete set.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Discrete set.

        Parameters:
        - *args, **kwargs: Any arguments and keyword arguments accepted by the
                           set constructor.
        """
        super().__init__(*args, **kwargs)
        # Calculate the number of elements in the set
        self.n = len(self)

    def __repr__(self) -> str:
        """
        Return a string representation of the Discrete set.

        Returns:
        - str: The string representation of the Discrete set.
        """
        return f"{super().__repr__()}"


class World(object):
    """
    Represents a virtual world environment for drone navigation and asset
    management.
    """
    def __init__(
            self, goal: tuple = (0, 0, 0), start: tuple = (5000, 5000, 5000),
            verbosity: int = 0, threshold: int = 40,
            size: tuple = (10000, 10000, 10000), ground: int = 0,
            step_length: int = 10, num_assets: int = 0,
            num_sensors: int = 0) -> None:
        """
        Initializes a World object with specified parameters.

        Parameters:
        - goal (tuple): The coordinates of the point of interest.
        - start (tuple): The coordinates of the starting point of the drone.
        - verbosity (int): The level of logging kept, 0 no logs kept, 1 the
                           the environment variables are logged (assets, size,
                           etc.), 2 the position of the drone at each step as
                           well as the environment variables.
        - threshold (int): The threshold cartesian distance of the drone to the
                           goal for the mission to be determined accomplished.
        - size (tuple): The three measurements defining the sides of the cube
                        that is the environment, (x, y, z).
        - ground (int): The z value defining where the ground is for assets
                        and collision detection.
        - step_length (int): The change in coordinates in any single axis for a
                             single action.
        - num_assets (int): The number of assets to create in the environment.
        - num_sensors (int): The number of ranging sensors that can detect
                             distance from objects, max 6.
        """
        # Assign arguments
        self.verbosity = verbosity
        self.threshold = threshold
        self.size = size
        self.ground = ground
        self.step_length = step_length
        self.num_assets = num_assets
        self.num_sensors = num_sensors
        # Get file ids for log file, position file, and new goal positions
        time = datetime.datetime.now()
        self.file_id = f'_{time.day}_{time.hour}_{time.minute}'
        # Define the world boundaries
        self.world_bounds = list(zip((0, 0, 0), self.size))
        self.x_range, self.y_range, self.z_range = [
            range(min_dim, max_dim) for min_dim, max_dim in self.world_bounds
        ]
        # Initialize point of interest
        self.goal = np.array(goal)
        # Initialize start point of drone
        self.start = np.array(start)
        # Initialize drone
        self.drone = Drone(start)
        # Create assets in environment
        self.assets = self._create_assets(self.num_assets)
        # Initialize state
        self.state = {
            "prev_position": self.start,
            "position": self.start,
            "goal": self.goal,
            "distance_to_goal": np.linalg.norm(self.goal - self.start)
        }
        self.state = State(self.state)
        # If there are sensors initialize the sensors
        if self.num_sensors:
            self._initialize_sensors()
        # Define action space
        self.action_space = Discrete({0, 1, 2, 3, 4, 5, 6})
        # Setup flight for the drone
        self._setup_flight()
        # Set mission completion flag to False
        self.goal_achieved = False
        # Log the world info
        if self.verbosity != 0:
            self._log_info()

    def _initialize_sensors(self) -> None:
        """
        Initializes sensors for the drone.

        This method creates sensor objects, scans the environment, and updates
        sensor readings in the drone's state.
        """
        # Initialize the sensors dictionary and sensor readings list in the
        # drone's state
        self.sensors = dict()
        self.state["sensor_readings"] = list()
        # Create sensors and update sensor readings
        for i in range(self.num_sensors):
            sensor_name = f"sensor_{i}"
            # Create a new Sensor object
            self.sensors[sensor_name] = Sensor(sensor_name, i)
            # Scan the environment using the sensor and add readings to the
            # sensor readings list
            self.state["sensor_readings"].append(
                self.sensors[sensor_name].scan(
                    self.assets["assets"],
                    self.state["position"],
                    self.ground
                )
            )

    def _update_sensor_readings(self) -> None:
        """
        Update sensor readings based on the drone's current position and
        detected assets.

        This method calculates distances between the drone and assets, checks
        if any assets are within sensing range, and updates sensor readings
        accordingly.
        """
        # Define the maximum distance at which assets may be detectable,
        # considering the maximum radius of the assets and range of sensor
        detect_dist = self.max_asset_radius + 40
        # Calculate distances between the drone and asset centers in 2D space
        dist = np.linalg.norm(
            self.assets['centers'][:, :2] - self.drone.position[:2],
            axis=1
        )
        # Check if any assets are within sensing range
        if any(dist < detect_dist):
            # Obtain the indices of the assets within that distance
            asset_ind = np.where(dist < detect_dist)
            # Update sensor readings for all sensors using detected assets
            self.state["sensor_readings"] = [
                self.sensors[sensor].scan(
                    self.assets["assets"][asset_ind],
                    self.state["position"],
                    self.ground
                ) for sensor in self.sensors
            ]

    def _log_info(self) -> None:
        """
        Log world data and asset information to a file.

        This method creates a log file containing information about the world,
        such as its size, step length, number of sensors, start and goal
        points, and asset centers with their heights and radii.
        """
        log_file = os.path.join(LOG_DIR, f'env_log_{self.file_id}.txt')
        content = f"""
        World Data:
        -----------------------------------------------
        World Size (x, y, z): {self.size}
        Step Length: {self.step_length}
        Start Point: {tuple(self.start)}
        Goal Point: {tuple(self.goal)}
        Number of Assets: {self.num_assets}
        -----------------------------------------------
        Asset Centers and Heights:
        """
        # Write info to file
        with open(log_file, 'w') as f:
            f.write(dedent(content))
            for asset in self.assets["assets"]:
                f.write(
                    ','.join(map(str, asset.center))
                    + f',{asset.height},{asset.radius}'
                    + '\n'
                )
        # Setup up position log file
        if self.verbosity > 1:
            self.position_file = os.path.join(
                LOG_DIR, f'position_log_{self.file_id}.txt'
            )
            # Keep a buffer to reduce file accessing
            self.buffer_size = 100
            self.position_buffer = deque(maxlen=self.buffer_size)
            self.position_buffer.append(tuple(self.drone.position))

    def _setup_flight(self) -> None:
        """
        Sets up the initial flight configuration for the drone.
        """
        self.drone.move_to_pos(self.start)

    def _get_obs(self) -> State:
        """
        Updates the state of the World by gathering observations.

        This method updates various attributes in the state dictionary based on
        the current state of the drone and environment, such as position,
        collision information, sensor readings, and goal achievement status.

        Returns:
        - State: The updated state dictionary containing observations.
        """
        # Update previous position to current position
        self.state['prev_position'] = self.state["position"]
        # Update current position of the drone
        self.state['position'] = self.drone.position
        # Update distance to goal
        self.state['distance_to_goal'] = np.linalg.norm(
            self.drone.position - self.goal
        )
        # If there are any sensors check there readings
        if self.num_sensors:
            self._update_sensor_readings()
        # Check for terminal states (collision and goal achieved)
        self.collision = self.drone.get_collision_info(
            self.assets["assets"],
            self.ground,
            self.world_bounds
        )
        self.goal_achieved = self.state['distance_to_goal'] < self.threshold

        return self.state

    def _do_action(self, action: int) -> None:
        """
        Executes an action for the drone.

        This method calculates the new position based on the given action and
        moves the drone accordingly.

        Parameters:
        - action (int): The action to be performed.
        """
        # Get current position components
        x0, y0, z0 = self.drone.position
        # Interpret the action to obtain the change in position
        x1, y1, z1 = self.interpret_action(action)
        # Move the drone to the new position
        self.drone.move_to_pos((x0 + x1, y0 + y1, z0 + z1))

    def _min_asset_radius(self) -> int:
        """
        Calculate the minimum radius for assets based on drone dimensions.

        This method calculates the minimum radius required for assets to be
        created in the environment based on the dimensions of the drone and
        the desired overlap between the drone's bounding box and the
        circumference of the asset, to reduce computation for collision
        calculations.

        Returns:
        - int: The minimum radius for assets.
        """
        # Obtain the max dimension of the drone in either the x or y axis
        max_dim = max([max-min for min, max in self.drone.bounds[:-1]])
        # Define the approximate max chord height (cm), this is the distance
        # overlap of the side of the drone and the circumference of the asset
        chord_height = 1
        # Calculate the minimum radius for assets to be created
        radius = int((chord_height**2 + (max_dim**2/4)) / (2*chord_height))

        return radius

    def _create_assets(self, num_assets: int) -> dict:
        """
        Creates assets and randomizes their parameters.

        Parameters:
        - num_assets (int): The number of assets to create.

        Returns:
        - dict: A dictionary containing the created assets and their centers.
        """
        # Define asset bounds
        min_radius = self._min_asset_radius()
        self.max_asset_radius = min_radius + 100
        radius_range = range(min_radius, self.max_asset_radius)
        height_range = range(300, 2000)
        # Obtain random points to place assets
        x_vals = random.sample(self.x_range, num_assets)
        y_vals = random.sample(self.y_range, num_assets)
        z_vals = [self.ground] * num_assets
        centers = np.array(list(zip(x_vals, y_vals, z_vals)))
        # Create random radii and heights for cylinders
        radii = random.sample(radius_range, num_assets)
        heights = random.sample(height_range, num_assets)
        # Create Cylinder objects using random parameters
        assets = [
            Cylinder(radius, center, height) for radius, center, height
            in zip(radii, centers, heights)
        ]

        return {'assets': np.array(assets), 'centers': centers}

    def _load_assets(self, asset_strings: list) -> dict:
        """
        Load assets from asset strings and create Cylinder objects.

        This method parses asset strings containing specifications for assets,
        creates Cylinder objects based on these specifications, and returns a
        dictionary containing the created assets and their centers.

        Parameters:
        - asset_strings (list): List of asset strings containing asset
                                specifications in the format
                                "center_x, center_y, center_z, height, radius".

        Returns:
        - dict: A dictionary containing the created assets as a numpy array
                under the key 'assets', and the centers of the assets as a
                numpy array under the key 'centers'.
        """
        # Cast asset specs to an array
        asset_specs = np.array([
            tuple(int(val) for val in asset.split(','))
            for asset in asset_strings
        ])
        # Return 0 length lists for assets and centers if there are no assets
        # to load
        if not asset_specs:
            return {'assets': [], 'centers': []}
        # Obtain asset parameters
        centers = asset_specs[:, :3]
        heights = asset_specs[:, 3]
        radii = asset_specs[:, 4]
        # Create assets
        assets = [
            Cylinder(radius, center, height) for radius, center, height
            in zip(radii, centers, heights)
        ]

        return {'assets': np.array(assets), 'centers': centers}

    def _compute_reward(self) -> tuple:
        """
        Computes the reward and termination status based on the current state.

        This method calculates the reward based on factors such as
        collisions and proximity to the target point. It also determines
        if the current state is terminal.

        Returns:
        - tuple: The computed reward and termination status
                 (0 for not finished, 1 for finished).
        """
        # Assign previous and current state positions
        prev_quad_pt = self.state["prev_position"]
        quad_pt = self.state["position"]
        # Assessing appropriate rewards for terminal states
        if self.collision:
            reward = -100
        elif self.goal_achieved:
            reward = 100
        else:
            # Initialize rewards
            reward = 0
            # Give a positive reward for moving closer to target point and
            # a negative reward for moving further from target point
            prev_dist = np.linalg.norm(prev_quad_pt - self.goal)
            dist = np.linalg.norm(quad_pt - self.goal)
            delta_dist = prev_dist - dist
            if delta_dist > 0:
                reward = delta_dist
            else:
                reward = delta_dist * 0.75
        # Determine if the state is terminal
        finished = 0
        if reward <= -10 or self.goal_achieved:
            finished = 1

        return reward, finished

    def step(self, action: int) -> tuple:
        """
        Perform one step in the environment.

        This method takes an action, updates the environment based on that
        action, gathers observations, computes the reward, and checks if the
        episode has terminated.

        Parameters:
        - action (int): The action taken by the agent.

        Returns:
        - tuple: A tuple containing the new observation (dict), the reward
                 received (int), and whether the episode has terminated
                 (int, 0 for not terminated, 1 for terminated).
        """
        # Perform the action in the environment
        self._do_action(action)
        # Get the updated observations from the environment
        obs = self._get_obs()
        # Compute the reward and check if the episode has finished
        reward, finished = self._compute_reward()
        # Determine if the episode has terminated
        done = 1 if finished == 1 else 0
        # Write positions to position_log
        if self.verbosity > 1:
            self.position_buffer.append(tuple(self.drone.position))
            if len(self.position_buffer) == self.buffer_size or done == 1:
                with open(self.position_file, 'a') as f:
                    for position in self.position_buffer:
                        f.write(','.join(map(str, position)) + '\n')
                    if done == 1:
                        f.write('\n\n')
                self.position_buffer.clear()

        return obs, reward, done

    def reset(self) -> dict:
        """
        Reset the environment to its initial state.

        This method sets up the initial flight configuration of the drone and
        obtains the initial observations.

        Returns:
        - dict: The initial observations of the environment.
        """
        # Set up the initial flight configuration of the drone
        self._setup_flight()
        # Obtain the initial observations of the environment
        obs = self._get_obs()

        return obs

    def interpret_action(self, action: int) -> tuple:
        """
        Interpret the action into a movement offset for the drone.

        Parameters:
        - action (int): The action taken by the agent.

        Returns:
        - tuple: The movement offset in (x, y, z) directions for the drone.
        """
        # Initialize the movement offset
        quad_offset = (0, 0, 0)
        # Map actions to movement offsets
        if action == 0:
            # Move right
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            # Move forward
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            # Move upward
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            # Move left
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            # Move backward
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            # Move downward
            quad_offset = (0, 0, -self.step_length)

        return quad_offset

    def action_space_sample(self) -> int:
        """
        Sample an action uniformly at random from the action space.

        Returns:
        - int: A randomly sampled action from the action space.
        """
        return random.choice(list(self.action_space))

    def render_path(self, episode: int | None = None) -> None:
        """
        Render the drone's trajectory and environment assets.

        This method generates an animation showing the drone's trajectory
        and the assets in the environment. It reads drone position data from
        a log file, extracts positions for the specified episode (or the last
        episode if no episode is specified), and animates the drone's movement.

        Parameters:
        - episode (int | None): The episode number to render. If None, the last
                                episode is rendered. If -1, the episode with
                                the most data is rendered. Defaults to None.
        """
        # If no position file output appropriate message
        if self.verbosity < 2:
            print(f"Verbosity: {self.verbosity}")
            print("No position log files created, cannot render.")
            return None

        # Function called for each animation step
        def update_position(i):
            # Get the point position from the environment
            x, y, z = next(iter_points, points[-1])
            # Append the new position to the trace
            trace_x.append(x)
            trace_y.append(y)
            trace_z.append(z)
            # Update the point and trace lines
            point.set_data([x], [y])
            point.set_3d_properties([z])
            line.set_data(trace_x, trace_y)
            line.set_3d_properties(trace_z)
            return point, line

        # Read in the drone position data from the file and split by episode
        data = []
        current_group = []
        with open(self.position_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    values = tuple(map(int, line.split(',')))
                    current_group.append(values)
                else:
                    if current_group:
                        data.append(current_group)
                        current_group = []
        # Add the last group to the data list if it's not empty
        if current_group:
            data.append(current_group)
        # Obtain only the positions for the episode of interest
        if episode is not None and 0 <= episode < len(data):
            points = data[episode]
        elif episode == -1:
            # The episode with the most data
            index = pd.Series([len(episode) for episode in data]).idxmax()
            points = data[index]
        else:
            # The last episode
            points = data[-1]
        # Iterable to find next position
        iter_points = iter(points)
        # Initialize the figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = self.start
        # Initialize the trace
        trace_x, trace_y, trace_z = [x], [y], [z]
        # Plot the initial point and trace
        point, = ax.plot(
            [x], [y], [z],
            marker='o',
            markersize=5,
            color='r',
            linestyle='None'
        )
        line, = ax.plot([], [], [], color='b')
        # Set axis limits
        ax.set_xlim([0, self.size[0]])
        ax.set_ylim([0, self.size[1]])
        ax.set_zlim([0, self.size[2]])
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Initialize the goal point
        goal_x, goal_y, goal_z = self.goal
        goal, = ax.plot(
            [goal_x], [goal_y], [goal_z],
            marker='x',
            markersize=10,
            color='g',
            linestyle='None'
        )
        # Plot assets in world
        for asset in self.assets['assets']:
            center_point = asset.center
            height = asset.height
            radius = asset.radius
            # Generate the cylinder mesh
            z = np.linspace(0, height, 100)
            theta = np.linspace(0, 2 * np.pi, 100)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = center_point[0] + radius * np.cos(theta_grid)
            y_grid = center_point[1] + radius * np.sin(theta_grid)
            # Plot the cylinder surface
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)
        # Animate
        ani = FuncAnimation(
            fig, update_position,
            frames=range(len(points)),
            interval=50,
            blit=True
        )
        plt.show()

    def render(self) -> None:
        """
        Render the drone's environment, including assets, start position, and
        goal position.

        This method creates a 3D plot using matplotlib to visualize the drone's
        environment. It plots the start position, goal position and assets.
        """
        # Initialize the figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the start position
        start_x, start_y, start_z = self.start
        ax.scatter(
            start_x, start_y, start_z, color='r', label='Start Position'
        )
        # Plot the goal position
        goal_x, goal_y, goal_z = self.goal
        ax.scatter(goal_x, goal_y, goal_z, color='g', label='Goal Position')
        # Plot assets in the environment
        for asset in self.assets['assets']:
            center_x, center_y, center_z = asset.center
            radius = asset.radius
            height = asset.height
            # Generate cylinder mesh for the asset
            z = np.linspace(0, height, 100)
            theta = np.linspace(0, 2 * np.pi, 100)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = center_x + radius * np.cos(theta_grid)
            y_grid = center_y + radius * np.sin(theta_grid)
            # Plot the cylinder surface for the asset
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)
        # Set axis labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        # Show the plot
        plt.show()

    def load_env(self, file_id: str) -> None:
        """
        Load environment settings and assets from a log file.

        This method loads environment settings and asset information from a log
        file created during a previous session. It parses the log file to
        obtain the world size, start position, goal position, and asset
        specifications, and then initializes the environment accordingly.

        Parameters:
        - file_id (str): The identifier of the log file to load.
        """
        # Set the file_id for loading the correct log file
        self.file_id = file_id
        # Obtain lines from environment log
        file = os.path.join(LOG_DIR, f'env_log__{file_id}.txt')
        with open(file, 'r') as f:
            lines = f.read().strip().split('\n')
        # Assign world size
        size = lines[2].split(': ')[1].strip('()').split(',')
        size = tuple(int(val) for val in size)
        # Assign start position
        start = lines[4].split(': ')[1].strip('()').split(',')
        start = tuple(int(val) for val in start)
        # Assign goal position
        goal = lines[5].split(': ')[1].strip('()').split(',')
        goal = tuple(int(val) for val in goal)
        # Number of assets
        self.num_assets = int(lines[6].split(': ')[1].strip())
        # If there are assets, create them
        if self.num_assets:
            # Define assets
            asset_strings = lines[9:]
            # Generate environment assets
            self.assets = self._load_assets(asset_strings)
            self.num_assets = len(self.assets['assets'])
        # Set verbosity level for logging
        self.verbosity = 2
        # Define the world boundaries
        self.size = size
        self.world_bounds = list(zip((0, 0, 0), self.size))
        self.x_range, self.y_range, self.z_range = [
            range(0, dim) for dim in self.size
        ]
        # Initialize goal and start positions
        self.goal = np.array(goal)
        self.start = np.array(start)
        # Initialize state
        _ = self.reset()
        # Define position log
        self.position_file = f'{LOG_DIR}position_log__{file_id}.txt'

    def get_random_goal(self, distance: float = 2000) -> tuple:
        """
        Generate a random goal point within a given distance from the start
        position.

        Parameters:
        - distance (float): The distance from the start position to the new
                            goal point. If None, a default distance of 2000
                            units is used.

        Returns:
         - tuple: The coordinates of the random goal point as a tuple
                  (x, y, z).
        """
        # Generate a random direction vector
        direction = np.random.uniform(-1, 1, size=3)
        # Normalize the direction vector
        direction /= np.linalg.norm(direction)
        # Generate a random point, distance units away from the start
        # along the random direction
        random_point = self.start + distance * direction
        random_point = tuple(int(val) for val in random_point)
        # Check if the random point is within the world boundaries, if not,
        # recursively find a new random point
        bounds = [self.x_range, self.y_range, self.z_range]
        if any([dim not in bound for dim, bound in zip(random_point, bounds)]):
            random_point = self.get_random_goal(distance)

        return random_point

    def update_goal(self, goal: tuple) -> None:
        """
        Update the goal position and log the new goal to a file.

        Parameters:
        - goal (tuple): The new goal position as a tuple (x, y, z).
        """
        # Update the goal position with the new goal tuple
        self.goal = np.array(goal)
        self.state['goal'] = self.goal
        # Log the new goal position to a file if verbosity is high enough
        if self.verbosity > 0:
            file_id = self.file_id
            file = os.path.join(LOG_DIR, f'goal_positions_{file_id}.txt')
            with open(file, 'a') as f:
                line_to_write = f'{goal}\n'
                f.write(line_to_write)
        # Reset the environment
        _ = self.reset()

    def assign_random_goal(self, distance_val: int | range) -> None:
        """
        Assign a random goal position within a given distance or range of
        distances from the start position.

        Parameters:
        - distance_val (int | range): If int, specifies the exact distance from
                                      the start position to the goal. If range,
                                      randomly chooses a distance within the
                                      range to set as the goal distance.
        """
        if isinstance(distance_val, int):
            goal = self.get_random_goal(distance_val)
        else:
            distance = random.choice(distance_val)
            goal = self.get_random_goal(distance)

        # Update the goal position with the new goal tuple
        self.update_goal(goal)


class Drone(object):
    """Represents a drone in a virtual environment."""

    def __init__(self, start: tuple) -> None:
        """
        Initialize a Drone object.

        Parameters:
        - num_sensors (int): The number of sensors on the drone.
        """
        # Set the default size of the drone (length, width, height)
        self.size = (20, 20, 5)
        # Set the initial position of the drone to the origin (0, 0, 0)
        self.position = np.array(start)
        # Calculate and set the initial boundaries of the drone
        self.bounds = self._get_bounds()

    def _get_bounds(self) -> list:
        """
        Calculate the lower and upper bounds for each axis based on the drone's
        size and position.

        Returns:
        - list: A list of tuples containing lower and upper bounds for each
                axis.
        """
        bounds = list()
        # For each axis (x, y, z), using the size of the drone, calculate the
        # corner positions of the hit box (bounding the drone) creating a
        # list of format [(x_lwr, x_upr), (y_lwr, y_upr), (z_lwr, z_upr)]
        for axis in range(0, 3):
            bound_length = self.size[axis] / 2
            lwr_bound = self.position[axis] - bound_length
            upr_bound = self.position[axis] + bound_length
            bounds.append((lwr_bound, upr_bound))

        return bounds

    def _range_overlap(self, range1: tuple, range2: tuple) -> bool:
        """
        Check if two ranges overlap.

        Parameters:
        - range1 (tuple): The first range tuple (start, end).
        - range2 (tuple): The second range tuple (start, end).

        Returns:
        - bool: True if the ranges overlap, False otherwise.
        """
        # Determine if two ranges overlap by comparing their start and end
        # points
        return max(range1[0], range2[0]) < min(range1[1], range2[1])

    def _range_contains(self, range1: tuple, range2: tuple) -> bool:
        """
        Check if range1 is completely contained within range2.

        Parameters:
        - range1 (tuple): The first range tuple (start, end).
        - range2 (tuple): The second range tuple (start, end).

        Returns:
        - bool: True if range1 is contained within range2, False otherwise.
        """
        # Check if range1's start point is greater than or equal to range2's
        # start point and range1's end point is less than or equal to range2's
        # end point
        return range2[0] <= range1[0] and range1[1] <= range2[1]

    def move_to_pos(self, new_position: tuple) -> None:
        """
        Move the drone to a new position specified by new_position.

        Parameters:
        - new_position (tuple): The new position coordinates (x, y, z).
        """
        self.position = np.array(new_position)

    def get_collision_info(
            self, world_assets: np.ndarray, ground: int,
            world_bounds: list) -> bool:
        """
        Check if the drone has collided with any objects in the environment or
        with the ground.

        Parameters:
        - world_assets (list): List of assets in the environment.
        - ground (float): Height of the ground.
        - world_bounds (list): Boundaries of the world.

        Returns:
        - bool: True if a collision is detected, False otherwise.
        """
        self.bounds = self._get_bounds()
        # Check if the drone has collided with the ground
        if self._range_overlap(self.bounds[2], (0, ground)):
            return True
        # Check if drone is still within the world bounds
        for drone_bound, world_bound in zip(self.bounds, world_bounds):
            if not self._range_contains(drone_bound, world_bound):
                # If any x, y, or z bound is not contained within the world
                # bound, there has been a collision
                return True
        # Define combinations of upper and lower bounds defining the hit box of
        # the drone for the x and y position
        combinations = list(itertools.product([0, 1], repeat=2))
        # Check each asset defined in the environment for collisions
        for asset in world_assets:
            asset_xy = asset.center[:-1]
            asset_height = (ground, ground + asset.height)
            # Check for each corner of the hitbox whether there is a collision
            for combo in combinations:
                i, j = combo
                drone_xy = (self.bounds[0][i], self.bounds[1][j])
                distance = np.linalg.norm(drone_xy - asset_xy)
                if (distance < asset.radius
                        and self._range_overlap(self.bounds[2], asset_height)):
                    return True

        return False


class Sensor(object):
    """A class representing a sensor attached to a drone."""

    def __init__(self, name: str, orientation: int) -> None:
        """
        Initializes a Sensor object with a specified name and orientation.

        Parameters:
        - name (str): The name of the sensor.
        - orientation (int): The orientation of the sensor, which determines
                             its direction and sign.
                             0: Positive X-axis
                             1: Positive Y-axis
                             2: Positive Z-axis
                             3: Negative X-axis
                             4: Negative Y-axis
                             5: Negative Z-axis
        """
        self.name = name
        # Default range of sensor
        self.range = (0, 40)
        # Determine the sign and axis based on the orientation
        self.sign, self.axis = {
            0: (1, 'x'),   # Positive X-axis
            1: (1, 'y'),   # Positive Y-axis
            2: (1, 'z'),   # Positive Z-axis
            3: (-1, 'x'),  # Negative X-axis
            4: (-1, 'y'),  # Negative Y-axis
            5: (-1, 'z')   # Negative Z-axis
        }[orientation]

    def _range_overlap(self, range1: tuple, range2: tuple) -> bool:
        """
        Check if two ranges overlap.

        Parameters:
        - range1 (tuple): The first range tuple (start, end).
        - range2 (tuple): The second range tuple (start, end).

        Returns:
        - bool: True if the ranges overlap, False otherwise.
        """
        # Determine if two ranges overlap by comparing their start and end
        # points
        return max(range1[0], range2[0]) < min(range1[1], range2[1])

    def _in_circle(self, xy_pos: tuple, center: tuple, radius: float) -> bool:
        """
        Check if a point is inside a circle.

        Parameters:
        - xy_pos (tuple): The coordinates of the point as a tuple (x, y).
        - center (tuple): The center of the circle as a tuple (h, k).
        - radius (float): The radius of the circle.

        Returns:
        bool: True if the point is inside the circle, False otherwise.
        """
        # Unpack the coordinates and center of the circle
        x0, y0 = xy_pos
        h, k = center
        # Calculate the squared distance from the point to the center of the
        # circle
        distance_squared = (x0 - h)**2 + (y0 - k)**2
        # Check if the squared distance is less than or equal to the square of
        # the radius
        return distance_squared <= radius**2

    def _in_rectangle(
            self, xy_pos: tuple, x_bounds: tuple, y_bounds: tuple) -> bool:
        """
        Check if a point is inside a rectangle.

        Parameters:
        - xy_pos (tuple): The coordinates of the point as a tuple (x, y).
        - x_bounds (tuple): The x-axis bounds of the rectangle as a tuple
                            (min_x, max_x).
        - y_bounds (tuple): The y-axis bounds of the rectangle as a tuple
                            (min_y, max_y).

        Returns:
        - bool: True if the point is inside the rectangle, False otherwise.
        """
        # Unpack the coordinates and bounds of the rectangle
        x0, y0 = xy_pos
        min_x, max_x = x_bounds
        min_y, max_y = y_bounds
        # Check if the point is within the x and y bounds of the rectangle
        return min_x <= x0 <= max_x and min_y <= y0 <= max_y

    def scan_pos(
            self, asset: Cylinder, drone_pos: np.ndarray,
            ground: float) -> float:
        """
        Determines the sensor reading based on the drone's position and the
        asset's characteristics.

        Parameters:
        - asset (Cylinder): The asset being scanned.
        - drone_pos (np.ndarray): The current position of the drone.
        - ground (float): The ground level in the environment.

        Returns:
        - float: The distance to the asset if detected, 0 otherwise.
        """
        # Determine the index for the relevant dimension based on the sensor's
        # axis
        range_i = {'x': 0, 'y': 1, 'z': 2}[self.axis]
        # Determine the index for distance calculation based on the sensor's
        # sign
        dist_i = {1: 0, -1: 1}[self.sign]
        # Determine the bounds based on the sensor's sign
        bounds = {1: self.range, -1: self.range[::-1]}[self.sign]
        # Define order for slicing based on the sensor's sign
        order = {1: slice(0, 2), -1: slice(1, -3, -1)}[self.sign]

        # Calculate the sensor's range based on its position and bounds
        sensor_range = tuple(
            drone_pos[range_i] + self.sign * bound for bound in bounds
        )
        radius = asset.radius
        center = asset.center

        # Define asset bounds for each dimension
        asset_x = (center[0] - radius, center[0] + radius)
        asset_y = (center[1] - radius, center[1] + radius)
        asset_z = (ground, ground + asset.height)
        h, k = center[:-1]

        # Return 0 if no detection is made
        detected_dist = 0
        # Check if the sensor axis is 'x' and the point is within the asset's
        # y-z rectangle
        if (self.axis == 'x'
                and self._in_rectangle(drone_pos[1:], asset_y, asset_z)):
            asset_x = (
                h + self.sign * (radius**2 - (drone_pos[1] - k)**2) ** (1 / 2),
                h
            )[order]
            # If the asset boundary is within detectable distance obtain the
            # reading
            if self._range_overlap(asset_x, sensor_range):
                detected_dist = abs(sensor_range[dist_i] - asset_x[dist_i])

        # Check if the sensor axis is 'y' and the point is within the asset's
        # x-z rectangle
        elif (self.axis == 'y' and
              self._in_rectangle(drone_pos[::2], asset_x, asset_z)):
            asset_y = (
                k + self.sign * (radius**2 - (drone_pos[0] - h)**2) ** (1 / 2),
                k
            )[order]
            if self._range_overlap(asset_y, sensor_range):
                detected_dist = abs(sensor_range[dist_i] - asset_y[dist_i])

        # Check if the sensor axis is 'z' and the point is within the asset's
        # circle
        elif (self.axis == 'z' and
              self._in_circle(drone_pos[:-1], center[:-1], radius)):
            if self._range_overlap(asset_z, sensor_range):
                detected_dist = abs(sensor_range[dist_i] - asset_z[dist_i])
        # Check if there are mutliplte readings for an asset and return min
        if isinstance(detected_dist, list):
            detected_dist = min(detected_dist)

        return detected_dist

    def scan(
            self, assets: np.ndarray, drone_pos: np.ndarray,
            ground: float) -> int:
        """
        Scans the environment for sensor readings based on drone position and
        assets.

        Parameters:
        - assets (list): List of assets in the environment.
        - drone_pos (np.ndarray): Current position of the drone.
        - ground (float): Ground level in the environment.

        Returns:
        - int: Minimum sensor reading for each asset that is detected, or 0 if
               no assets are detected.
        """
        # Generate sensor readings for each asset
        readings = [
            self.scan_pos(asset, drone_pos, ground)
            for asset in assets
        ]
        # Filter out zero readings
        sensor_readings = [val for val in readings if val != 0]
        # Return sensor readings if available, otherwise return 0
        return min(sensor_readings) if sensor_readings else 0

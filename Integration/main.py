import requests
import os
import ast
import time
import argparse
import pymavlink_SITL as sitl
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, filename='sar_drone.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


# read in config file
def read_config_file(config_file_path: str) -> dict:
    # check to see if file exist
    if os.path.exists(config_file_path):
        config_data = {}

        # read in config
        with open(config_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):  # ignore comments
                    key, value = line.split('=', 1)
                    config_data[key] = value

        logging.info(f"Config File: {config_data}")
        return config_data
    else:
        logging.warning(f"Configuration file {config_file_path} does not exist.")


# Flask API preparation
def set_start(start: tuple, goal: tuple) -> None:
    """
    Set the start and goal positions for the mission.

    Sends a request to the server API to assign the start and goal
    positions for the mission. Prints a success message if the request is
    successful; otherwise, prints an error message.

    Parameters:
    - start (tuple): Tuple containing the start position coordinates
                     (lat, lon, alt).
    - goal (tuple): Tuple containing the goal position coordinates
                    (lat, lon, alt).
    """
    logging.debug(f"DQN Start from: {start}")
    logging.debug(f"DQN Go to: {goal}")

    # Construct the URL with start and goal parameters
    extension = '/assign-start-goal?start={}_{}_{}&goal={}_{}_{}'
    url = configs['api_server'] + configs['DQN_port'] + extension.format(*start, *goal)
    logging.debug(f"Main DQN Request: {url}")

    # Send a GET request to the server API
    response = requests.get(url)
    # Check the response status code
    if response.status_code == 200:
        logging.info("Set start successfully!")
    else:
        logging.info("Error while tryig to start!")


# Request path from DQN server API
def get_path_from_api() -> str:
    """
    Retrieve the path information from the API endpoint.

    Makes a GET request to the server API to retrieve the path information.
    Returns the path content if the request is successful; otherwise, prints
    an error message and returns None.

    Returns:
    - str: Path content retrieved from the API.
    """
    # Make a GET request to the API endpoint
#    response = requests.get(api_server + DQN_port + '/get-path')
    response = requests.get(configs['api_server'] + configs['DQN_port'] + '/get-path') 

    # Check if the request was successful (status code 200)
    if response.status_code == 201:
        # Extract the path content from the response
        path_content = response.text
        return path_content
    else:
        # Handle any errors or unexpected status codes
        logging.error('DQN Path Request Failed! Error Code:', response.status_code)
        return None


# Call the DQN Flask API
def call_DQN_API(start, goal):
    # Call the DQN
    set_start(start, goal)
    # Call the function to get path content from the API
    path_content = get_path_from_api()
    # Check if path content was retrieved successfully
    if path_content is not None:
        return path_content
    else:
        logging.error('Failed to retrieve DQN path.')
        raise Exception('DQN path planning failed!')


# Call the A* FastAPI
def call_astar_endpoint(start, goal, obstacles):
#    url = api_server + AStar_port + '/astar/'
    url = configs['api_server'] + configs['Astar_port'] + '/astar/'
    payload = {
        'start': start,
        'goal': goal,
        'obstacles': obstacles,
    }
    logging.info(f"A* API: {url}")
    logging.info(f"A* Request: {payload}")

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        logging.info("A* mission received OK")
        return response.json()
    else:
        logging.error("A* Route Request Failed! Status Code:", response.status_code)
        raise Exception('A* path planning failed!')

def call_rrt_endpoint(start, goal, obstacles):
#    url = api_server + RRT_port + '/rrt/'
    url = configs['api_server'] + configs['RRT_port'] + '/rrt/'
    payload = {
        'start': start,
        'goal': goal,
        'obstacles': obstacles
    }
    logging.info(f"RRT API: {url}")
    logging.info(f"RRT Request: {payload}")

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        logging.info("RRT mission plan received OK")
        return response.json()
    else:
        logging.error("RRT Route Request Failed! Status Code:", response.status_code)
        raise Exception('RRT path planning failed!')


# main module for the SAR drone program
def main():
    """
    Main function get path from DQN Network, A* algorithm and RRT algorithm

    Retrieves start and goal coordinates from command-line arguments or uses
    default values. It then calls API to get mission plans and execute it.
    """

    # Read Config File
    global configs
    configs = read_config_file("config.sar_drone")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='commands')
#    parser.add_argument('--help')

    # use mission file or REST API mission query, but not both
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument('--mission')
    cmd_group.add_argument('--model')

    args = parser.parse_args()

    # Connection configs
    mission_file = args.mission
    path_planning_type = args.model
    connection_string = configs["transport"] + configs["sim_server"] + configs["sim_port"]

    logging.info(f"Connection String: {connection_string}")
    logging.info(f"Mission File: {mission_file}")
    logging.info(f"Path Planning Type: {path_planning_type}")

    # start and goal
    start = ast.literal_eval(configs['start'])
    goal = ast.literal_eval(configs['goal'])
    obstacles = ast.literal_eval(configs['obstacles'])

    # Time the algorithms
    start_time = time.time()

    # Generate Mission Plans
    if mission_file:
        mission_plan = sitl.read_mission_file(mission_file)
        print(mission_plan)
    else:
        if path_planning_type == "DQN":
            # Request DQN mission plan
            mission_plan = call_DQN_API(start, goal)
            print(mission_plan)
        elif path_planning_type == "RRT":
            # Request RRT mission plan
            mission_plan = call_rrt_endpoint(start, goal, obstacles)
            [print(wp) for wp in mission_plan]
        elif path_planning_type == "A*":
            # Request A* mission plan
            print("Sorry! A* Algorithm NOT implemented yet")
#            mission_plan = call_astar_endpoint(start, goal, obstacles)
        else:
            print(f"ERROR: Unknown Path Planning Type {path_planning_type}")
            logging.error(f"ERROR: Unknown Path Planning Type {path_planning_type}")
            raise Exception('Path planning type not supported')

    # Display time used to generate Mission Plan
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Mission Plan Generated in {execution_time:.3f} sec using {path_planning_type}")

    # Execute Mission
    start_time = time.time()
    sitl.run_mission(connection_string, mission_plan)
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Mission Executed in {execution_time:.3f} sec using {path_planning_type}")


if __name__ == '__main__':
    main()

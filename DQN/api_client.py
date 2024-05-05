import requests
import argparse

# Define the base URL of server API
base_url = 'http://127.0.0.1:5000'


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


def parse_opt() -> argparse.ArgumentParser:
    """
    Parse command line options for setting up the drone's starting and goal
    positions.

    Returns:
    - argparse.ArgumentParser: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start', type=parse_tuple, default=(42.355118, -71.071305, 6),
        help='The starting position of the drone, latitude_longitude_altitude'
    )
    parser.add_argument(
        '--goal', type=parse_tuple, default=(42.355280, -71.070911, 26.5),
        help='The goal position of the drone, latitude_longitude_altitude'
    )

    return parser.parse_args()


def set_start_and_goal(start: tuple, goal: tuple) -> None:
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
    # Construct the URL with start and goal parameters
    extension = '/assign-start-goal?start={}_{}_{}&goal={}_{}_{}'
    url = base_url + extension.format(*start, *goal)
    # Send a GET request to the server API
    response = requests.get(url)
    # Check the response status code
    if response.status_code != 200:
        print('Error: Could not set start and goal positions')


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
    response = requests.get(base_url + '/get-path')
    # Check if the request was successful (status code 200)
    if response.status_code == 201:
        # Extract the path content from the response
        path_content = response.text
        return path_content
    else:
        # Handle any errors or unexpected status codes
        print('Error:', response.status_code)
        return None


def main(opt: argparse.ArgumentParser) -> None:
    """
    Main function to demonstrate API usage.

    Retrieves start and goal coordinates from command-line arguments or uses
    default values. Calls API functions to set start, get path, and process
    path content.
    """
    # Call the function to set start and goal coordinates using the API
    set_start_and_goal(opt.start, opt.goal)
    # Call the function to get path content from the API
    path_content = get_path_from_api()
    # Check if path content was retrieved successfully
    if path_content is not None:
        # Process or use the path content as needed
        print(path_content)
    else:
        print('Failed to retrieve path content from API.')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

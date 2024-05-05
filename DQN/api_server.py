from flask import Flask, request, make_response, Response
from path_planner import PathPlanner

# Create a Flask app
app = Flask(__name__)

# Create an instance of Model
planner = PathPlanner()


@app.route('/assign-start-goal')
def set_start_goal() -> Response:
    """
    Define a route for assigning start and goal positions.

    Returns:
    - Response: HTTP response indicating success or providing a message with
                default positions.
    """
    # Get start and goal positions from the request arguments
    start_position = request.args.get('start')
    goal_pos = request.args.get('goal')
    # If start or goal positions are not provided, return a message with the
    # default positions
    if not start_position or not goal_pos:
        msg = '/assign-start-goal?start={}_{}_{}&goal={}_{}_{}'
        return msg.format(*planner.geo_start, *planner.geo_goal)
    # Parse start and goal positions from the request arguments
    start = [float(val) for val in start_position.split('_')]
    goal = [float(val) for val in goal_pos.split('_')]
    # Set the start and goal positions in the model
    planner.set_start(*start)
    planner.set_goal(*goal)
    # Create a response indicating success
    response = make_response()
    response.status_code = 200

    return response


@app.route('/get-path')
def get_path() -> tuple:
    """
    Define a route for retrieving the path from the model.

    This route sets up the mission in the model, executes actions until
    completion, and retrieves the resulting path.

    Returns:
    - tuple: A tuple containing the path and the HTTP status code 201.
    """
    # Get the resulting path from the model
    path = planner.run()

    # Return the path and HTTP status code 201
    return path, 201


def get_base_url():
    with app.test_request_context():
        base_url = request.base_url

    return base_url


if __name__ == '__main__':
    app.run(debug=True)

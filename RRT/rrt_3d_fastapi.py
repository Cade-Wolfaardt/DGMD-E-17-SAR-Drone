# Functions for path planning using the Rapidly-exploring Random Tree
# or RRT probablistic algorithm

from fastapi import FastAPI
from urllib.parse import unquote
from typing import List, Tuple
import numpy as np
import random as rd
import path_planning_utils as ppu
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, filename='sar_drone.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


# FastAPI instance
app = FastAPI()


# Steer in 3D with geo locations, notice y(lat) comes before x(lon)
def steer(from_node: ppu.Node, to_point, step_size) -> ppu.Node:
    # use geo coorindates
    from_point = from_node.point
    dy, dx, dz = np.array(to_point) - np.array(from_point)

    if dx == 0 and dy == 0:
        # A vertical jump
        to_point = (from_point[0], from_point[1], min(step_size, dz))
        new_node = ppu.Node(to_point)
    else:
        # az = calculate_azimuth(dy, dx)
        az, dist = ppu.llh_to_meters(from_point, to_point)
        step_length = min(step_size, dist)
        new_node = ppu.get_node(from_node, az, step_length, dz)

    new_node.parent = from_node
    return new_node


# build out the path
def build_rrt_3d(start, goal, step_size, max_iterations, obstacles):
    start_node = ppu.Node(start)
    goal_node = ppu.Node(goal)
    nodes = [start_node]

    # bounds
    lat_min = min(start[0], goal[0])
    lat_max = max(start[0], goal[0])
    lon_min = min(start[1], goal[1])
    lon_max = max(start[1], goal[1])
    alt_min = 5
    alt_max = 100  # max altitude for the drone flight

    for _ in range(max_iterations):
        rand_point = (rd.uniform(lat_min, lat_max), rd.uniform(lon_min, lon_max), rd.uniform(alt_min, alt_max))
        nearest = ppu.nearest_node(nodes, rand_point)
        new_node = steer(nearest, rand_point, step_size)

        if new_node and not ppu.is_collision(new_node.point, obstacles):
            nodes.append(new_node)
            # almost reached goal?
            if ppu.llh_to_meters(new_node.point, goal)[1] < step_size:
                goal_node.parent = new_node
                nodes.append(goal_node)
                break
    return nodes


def reconstruct_path(nodes, start_node, goal_node):
    path = []
#   current_node = goal_node
    current_node = nodes[-1]
    while current_node != start_node and current_node is not None:
        path.append(current_node.point)
        current_node = current_node.parent
    path.append(start_node.point)
    path.reverse()
    return path


@app.post("/rrt")
#async def generate_rrt_path(start: Tuple[float, float, float], goal: Tuple[float, float, float], obstacles: List[Tuple[Tuple[float, float, float], float]]):
async def generate_rrt_path(start, goal, obstacles):
    # Settings for RRT
    step_size = 1  # 1 meter
    max_iterations = 1000

    # process params
    print(f"input query strings {start}, {goal}, {obstacles}")
    s = unquote(start)
    g = unquote(goal)
    o = unquote(obstacles)
    print(f"URL decoded {s}, {g}, {o}")
    start = tuple(s)
    goal = tuple(g)
    obstacles = list(tuple(tuple(o)))

    # Build the RRT and get nodes
    nodes = build_rrt_3d(start, goal, step_size, max_iterations, obstacles)

    # Extract path from the node connections
    path = reconstruct_path(nodes, ppu.Node(start), ppu.Node(goal))

    mission_plan = ppu.path_to_mission(path)

    return mission_plan

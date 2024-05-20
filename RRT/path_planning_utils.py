# Utility functions shared by path planning methods
#

import numpy as np
from math import sqrt
from pyproj import Geod

# Geo distance
geod = Geod(ellps="WGS84")


class Node:
    def __init__(self, point):
        self.point = point
        self.parent = None


# get distance and bearing in 3D between two geo coordinates + altitude
def llh_to_meters(point1, point2):
    az12, az21, dist = geod.inv(point1[1], point1[0], point2[1], point2[0])
    return az12, sqrt(dist**2 + (point1[2] - point2[2])**2)


# get azimuth/bearing angle
def calculate_azimuth(x, y):
    azimuth = np.arctan2(y, x) * (180 / np.pi)
    azimuth = (90 - azimuth) % 360  # as measured from the true North
    return azimuth


# get the lon/lat/alt of the a new node from an existing node
def get_node(from_node: Node, azimuth, length, alt_change) -> Node:
    lon, lat, _ = geod.fwd(from_node.point[1], from_node.point[0], azimuth, length)
    alt = from_node.point[2] + alt_change
    return Node((lat, lon, alt))


# Check for collision with obstacles
def is_collision(point, obstacles):
    # Check spherical obstacles
    for center, radius in obstacles:
        if llh_to_meters(point, center)[1] <= radius:
            return True
    return False


def nearest_node(nodes, point):
    return min(nodes, key=lambda node: llh_to_meters(node.point, point)[1])


# take a path and convert to QGC mission format
def path_to_mission(path):
    # default values
    take_off_altitude = 5
    wp_type = 82  # Waypoint type: 16=waypoint, 82=spline waypoint
    mission = []

    # Output the path in QGC WPL 110 format
    mission.append("QGC WPL 110")

    # take-off command
    #print(f"0\t1\t0\t22\t0\t0\t0\t0\t{path[0][0]:.6f}\t{path[0][1]:.6f}\t{take_off_altitude}\t1")
    mission.append(f"0\t1\t0\t22\t0\t0\t0\t0\t{path[0][0]:.6f}\t{path[0][1]:.6f}\t{take_off_altitude}\t1")

    # Waypoints
    for index, point in enumerate(path, start=1):
    #    print(f"{index}\t0\t3\t{wp_type}\t0\t0\t0\t0\t{point[0]:.6f}\t{point[1]:.6f}\t{point[2]:.1f}\t1")
        mission.append(f"{index}\t0\t3\t{wp_type}\t0\t0\t0\t0\t{point[0]:.6f}\t{point[1]:.6f}\t{point[2]:.1f}\t1")

    # RTL landing
    #print(f"{len(path)+1}\t0\t3\t20\t0\t0\t0\t0\t{path[0][0]:.6f}\t{path[0][1]:.6f}\t{path[0][2]:.1f}\t1")
    mission.append(f"{len(path)+1}\t0\t3\t20\t0\t0\t0\t0\t{path[0][0]:.6f}\t{path[0][1]:.6f}\t{path[0][2]:.1f}\t1")

    return mission

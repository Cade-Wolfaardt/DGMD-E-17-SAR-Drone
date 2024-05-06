# Server Architecture Visualization

This Python script uses the Manim library to create an animated visualization of a server architecture involving path planning, simulation, and drone flight management. It showcases different server endpoints, their interactions, and the dynamic behavior of systems through animations.

## Description

The animation is structured into three main parts:
- **Path Planning**: Displays different algorithms like DQN and RRT.
- **Simulation**: Shows interaction between simulation components like pymavlink, ArduPilot, and QGroundControl.
- **Drone Flight**: Visualizes the components involved in drone flights such as PixHawk, MAVSDK, and DJI Mini 2.

Each component is visually separated using boxes, and interactions are demonstrated with moving packets representing data flow.

## Installation

Before running the animation, ensure you have Manim installed. Manim can be installed via pip:

```pip install manim

## Usage
To run the animation, use the following command:

```manim -p -ql data_flow_animation.py ServerArchitecture

This command will render the scene in low quality for quick preview. For higher quality, replace -ql with -qh.

##Files
data_flow_animation.py: Contains the Manim script for the animation.
drone.svg: Contains vector graph of the drone used in the animation.
requirements.txt: required library

##Contributing
Contributions to this script are welcome! Please fork the repository and submit a pull request with your additions.

##License
This project is released under the MIT License. See the LICENSE file for more details.

##Acknowledgements
Thanks to the Manim community for providing an excellent tool for mathematical and system visualizations.

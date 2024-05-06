# DGMD-E-17 SAR Drone Integration

This repository contains the integration code for a Search and Rescue (SAR) drone, part of the DGMD E-17 course project. The project aims to demonstrate the practical application of autonomous drones in SAR operations.

## Components

- `main.py`: The main script that orchestrates path plan generation and SITL simulation
- `config.sar_drone`: Configuration settings for the drone.
- `pymavlink_SITL.py`: Interface for drone simulation testing.

## Setup

Ensure you have the required dependencies installed by running:

pip install -r requirements.txt

## Usage
Run the main script to generate mission plan and start the drone simulation:

python main.py --model <"DQN" | "RRT">

DQN - Generate mission plan from the Deep Q-Network Flask server then run simulation in ArduPilot
RRT - Generate mission plan from the Rapidly-exploring Random Tree FastAPI server and run simulation in ArduPilot

## Contributing
Contributions to improve the functionality or efficiency of the SAR drone are welcome. Please fork this repository and submit your pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

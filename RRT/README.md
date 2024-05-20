# RRT Path Planning for SAR Drone

This section of the DGMD-E-17 project repository focuses on the implementation of the Rapidly-exploring Random Tree (RRT) algorithm, which is used for path planning in search and rescue (SAR) drone operations.

## Overview

The RRT algorithm helps in navigating complex environments by efficiently searching non-uniform spaces, making it ideal for SAR missions where quick and safe navigation is critical.

## Files

- `rrt.py`: Contains the RRT algorithm implementation.
- `utils.py`: Helper functions that support the RRT algorithm.

## Requirements

Ensure you have the necessary Python packages installed:

pip install -r requirements.txt

## Usage
To run the RRT FastAPI Server

	uvicorn rrt_3d_fastapi:app --host 0.0.0.0 --port 5020 --reload


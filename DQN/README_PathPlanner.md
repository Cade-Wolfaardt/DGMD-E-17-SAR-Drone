### PathPlanner

The `PathPlanner` class represents a mission path planner, incorporating the environment, the trained DQN model, and the desired output for mission paths.

### Usage

Simply obtaining a path:
```python
# Create an instance of the PathPlanner class
planner = PathPlanner()

# Generate the mission path
path = planner.run()
```
The path is now stared in the `path` variable and usable in script.

Outputting a path to a txt file and rendering:
```python
# Create an instance of the PathPlanner class defining the need to render
# in the simulation environment after completion and output the mission path
# to a txt file
planner = PathPlanner(render=True, verbosity=2, to_txt=True)

# Generate the mission path
path = planner.run()
```
_Note: When rendering verbosity needs to be set to 2 as the rendered path is first saved to a log file and then loaded upon render._

## Constants

`NETWORKS` is a defined dictionary storing the several different trained models, using their prefix as the key and the number of hidden layers and the associated neurons.

## Script

The `path_planner.py` script houses a `main` function that can parse system arguments making it easy to explore the `PathPlanner` class when executing the script.

When executing the script several arguments can be passed:
```
usage: path_planner.py [-h] [--id ID] [--model_dir MODEL_DIR] [--neurons NEURONS] [--layers LAYERS]
                       [--verbosity VERBOSITY] [--render RENDER] [--to_txt TO_TXT] [--step_length STEP_LENGTH]
                       [--threshold THRESHOLD] [--delta_dist DELTA_DIST] [--start START] [--goal GOAL]
                       [--mission_dir MISSION_DIR] [--to_csv TO_CSV] [--cartesian_start CARTESIAN_START]
```

Every argument is detailed in the `PathPlanner` class; further, `help` messages are available.

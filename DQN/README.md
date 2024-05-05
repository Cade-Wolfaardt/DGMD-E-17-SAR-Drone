## Getting Started

To ensure a clean and isolated environment, setting up a virtual environment is recommended before running the files. Execute the following command in the command line to install the required dependencies:
```
pip install -r requirements.txt
```

## File Structure
```bash
DQN
├───api_client.py
├───api_server.py
├───environment.py
├───eval.py
├───path_planner.py
├───README.md
├───requirements.txt
├───train.py
├───utils.py
│
├───Logs
│   ├───env_log__5_5_28.txt
│   └───position_log__5_5_28.txt
│
├───Missions
│   ├───final_mission_file.csv
│   └───final_mission_file.txt
│
└───Models
    ├───final_DQN_policy_net.pth
    ├───README.md
    ├───_23_22_5_policy_net.pth
    ├───_24_16_52_policy_net.pth
    ├───_24_19_41_policy_net.pth
    ├───_24_21_57_policy_net.pth
    ├───_25_17_48_policy_net.pth
    ├───_25_21_5_policy_net.pth
    ├───_26_23_6_policy_net.pth
    └───_27_16_10_policy_net.pth
```
### Files
* __api_client.py__ - Demonstrates how to access and use the server-side API.
* __api_server.py__ - Implements the server-side API for the path planner.
* __environment.py__  - Contains the `World `class for simulating, training, and testing of models.
* __eval.py__ - Includes the `Eval` class for testing and evaluating different trained DQN models.
* __path_planner.py__ - CHouses the `PathPlanner` class responsible for generating mission paths.
* __train.py__ - Used for training new DQN models.
* __utils.py__ - Holds utility items like training constants, the `DQN` class, the `DQN3` class, `ReplayMemory` class, and `Transition` class.

### Folders
* __Logs__ - Stores log files generated during simulation for rendering and loading previous environments.
* __Missions__ - Contains mission path files in QGC format for drone control or simulation, and CSV files for plotting routes.
* __Models__ - Holds trained DQN models, with `final_DQN_policy_net.pth` representing the best model trained.

## Training

The `train.py` script can be run locally or on Google Colab (transferring the code to an ipynb is recommended).

## Functions

* `select_action` is implements an epsilon greedy algorithm during training so and facilitates action choice.

* `optimize_model` optimizes the DQN network during the training process.

* `plot_steps` a helper function used to visualize training progress.

* `main` houses the entire training loop, and handle declaration of the environment, the number of episodes during training, etc.

## Usage

If running locally:
```
python train.py
```
_Note: The model automatically set the directory for where to save models and logs, so ensure if training online or locally that the directory exists_

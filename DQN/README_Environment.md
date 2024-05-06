# Classes

## Cylinder

The `Cylinder` class is designed to model cylinder-shaped assets within the simulation environment. It features three essential attributes: radius, center, and height. The center attribute represents the Cartesian coordinate of the cylinder's base center, height denotes the cylinder's height, and radius specifies the cylinder's radius.

#### Usage

Create a `Cylinder`:
```python
# Creating a Cylinder
cylinder = Cylinder(
  radius=10.5,
  center=(100, 150, 0),
  height=23.5
)
```

## State

The `State` class encapsulates a state dictionary that defines the environment's current state. It offers additional functionalities such as processing and filtering out unnecessary entries from the state dictionary. Moreover, it facilitates converting the dictionary into a `torch.Tensor`, which is essential for utilizing it within PyTorch models. This class relies on the `torch` library for converting the dictionary to a `torch.Tensor`.

#### Usage

Creating a `State`:
```python
# Creating an empty State
empty_state = State()

# Creating a State with one item
singular_state = State({'position': (0, 0, 0)})

# Casting multiple items to a State
multi_state = State({
  'position': (0, 0, 0),
  'next_positon': (0, 0, 10),
  'goal': (0, 0, 100)
})
```
_Note: State inherits from dict and thus can be used like a dictionary in creation, expanding, or removing._

Processing a state for use as a `torch.Tensor`:
```python
# Create a State
state = State({
  'position': (0, 0, 0),
  'next_position': (0, 0, 10),
  'goal': (0, 0, 100)
})
# Define keys to remove
rem_keys = ['next_position']
# Define device to process state
device = 'cpu'
# Process state and output representation
print(state.process(device, rem_keys))
```
_Note: State.process() generates a copy of the state and carries out the changes on the copy preserving the original state._

Output:
```
tensor([  0.,   0.,   0.,   0.,   0., 100.])
```

## Discrete

The `Discrete` class is a subclass of set and represents a discrete set of elements. It includes an additional attribute `n`, which signifies the number of elements in the set. This class plays a crucial role in defining the action space of the environment.

Creating a `Discrete` instance:
```python
# Creating an empty Discrete instance
empty_discrete = Discrete()

# Creating a Discrete instance
discrete = Discrete({0, 1, 2, 3, 4, 5, 6})

# Casting a set to Discrete
orginal_set = {1, 2, 3}
discrete_cast = Discrete(original_set)
```

Obtaining the size of the discrete set:
```python
# Create an action space as an instance of Discrete
action_space = Discrete({0, 1, 2, 3, 4, 5, 6})

# Obtain length
length = action_space.n

# Output results
print(action_space, length)
```

Output:
```
(Discrete({0, 1, 2, 3, 4, 5, 6}), 7)
```
## World

Creating a Random Environment and Visualizing It
```python
from environment import World
from utils import goal, start_position

env = World(goal, start_position, 2)
env.render()
```
This code snippet creates an environment with 60 assets randomly placed within it and visualizes the environment.

Loading an Environment:
```python
from environment import World

# Create a placeholder environment
env = World()
# Load environment from environment log with id 5_5_28
env.load_env('5_5_28')
```
This code snippet demonstrates how to load an environment from an environment log file with the specified ID. Please note that the environment log file should follow the naming convention 'env_log__XX_XX_XX.txt' for this operation to work correctly.

Rendering a Mission From a Position log:
```python
from environment import World

# Create a placeholder environment
env = World()
# Load environment from environment log with id 5_5_28
env.load_env('5_5_28')
# Render the mission path
env.render_path()
```
This code snippet demonstrates how to render a mission path from a position log using the `World` class. It loads the environment from the specified log and then renders the mission path for visualization.

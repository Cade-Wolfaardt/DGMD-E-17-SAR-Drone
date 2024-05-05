## Cylinder

The `Cylinder` class is designed to model cylinder-shaped assets within the simulation environment. It features three essential attributes: radius, center, and height. The center attribute represents the Cartesian coordinate of the cylinder's base center, height denotes the cylinder's height, and radius specifies the cylinder's radius.

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

## World

## Drone

## Sensor

## Cylinder

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

## World

## Drone

## Sensor

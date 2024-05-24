# Linen Applications

Linen applications (lipps) provides deep learning models built on Flax linen API.

## Features

- Pretrained variables
- Independent implementations

## Example

```python
import jax
import lipps
from lipps import models

# Create model.
model = models.movinet()

# Initialize variables.
rng = jax.random.PRNGKey(0)
variables = model.init(rng, jax.numpy.zeros((16, 224, 224, 3)), is_training=True)

# [Optional] Assign pretrained variables to the initialized variables.
pretrained = ...
variables, masks = lipps.assign_variables(variables, pretrained)
```

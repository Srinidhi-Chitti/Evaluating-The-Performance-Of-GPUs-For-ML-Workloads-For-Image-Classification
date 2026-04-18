# from flax import nnx
# from functools import partial
# from typing import Optional

# class CNN(nnx.Module):

#   def __init__(self, *, rngs: nnx.Rngs):
#     self.conv1 = nnx.Conv(1, 32, kernel_size=(3,3), rngs=rngs)
#     self.batch_norm1 = nnx.BatchNorm(32, rngs=rngs)
#     self.dropout1 = nnx.Dropout(rate=0.025)

#     self.conv2 = nnx.Conv(32, 64, kernel_size=(3,3), rngs=rngs)
#     self.batch_norm2 = nnx.BatchNorm(64, rngs=rngs)

#     self.avg_pool = partial(nnx.avg_pool, window_shape=(2,2), strides=(2,2))

#     self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
#     self.dropout2 = nnx.Dropout(rate=0.025)
#     self.linear2 = nnx.Linear(256, 10, rngs=rngs)

#   def __call__(self, x, rngs: Optional[nnx.Rngs] = None):

#     x = self.avg_pool(
#         nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs)))
#     )

#     x = self.avg_pool(
#         nnx.relu(self.batch_norm2(self.conv2(x)))
#     )

#     x = x.reshape(x.shape[0], -1)

#     x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
#     x = self.linear2(x)

#     return x
import jax
import jax.numpy as jnp
import flax.linen as nn


class CNN(nn.Module):
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x):

        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x,(2,2),(2,2))

        x = nn.Conv(features=64, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x,(2,2),(2,2))

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(128)(x)
        x = nn.relu(x)

        x = nn.Dense(self.num_classes)(x)

        return x


def initialize_model():

    model = CNN(num_classes=10)

    rng = jax.random.PRNGKey(0)

    dummy = jnp.ones((1,28,28,1))

    params = model.init(rng, dummy)['params']

    return model, params
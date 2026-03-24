# !pip install -U "jax[cuda12]"
# !pip install -U flax
import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

tf.random.set_seed(0)  # Set the random seed for reproducibility.

train_steps = 1200
eval_every = 200
batch_size = 32

train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

train_ds = train_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # normalize train set
test_ds = test_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # Normalize the test set.

# Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
train_ds = train_ds.repeat().shuffle(1024)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

from flax import nnx  # The Flax NNX API.
from functools import partial
from typing import Optional

class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.batch_norm1 = nnx.BatchNorm(32, rngs=rngs)
    self.dropout1 = nnx.Dropout(rate=0.025)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.batch_norm2 = nnx.BatchNorm(64, rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.dropout2 = nnx.Dropout(rate=0.025)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x, rngs: Optional[nnx.Rngs] = None):
    x = self.avg_pool(nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs))))
    x = self.avg_pool(nnx.relu(self.batch_norm2(self.conv2(x))))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
    x = self.linear2(x)
    return x

# Instantiate the model.
model = CNN(rngs=nnx.Rngs(0))
# Visualize it.
nnx.display(model)

import jax.numpy as jnp  # JAX NumPy

y = model(jnp.ones((1, 28, 28, 1)), nnx.Rngs(0))
y

import optax

learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(
  model, optax.adamw(learning_rate, momentum), wrt=nnx.Param
)
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)

nnx.display(optimizer)

def loss_fn(model: CNN, rngs: nnx.Rngs, batch):
  logits = model(batch['image'], rngs)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, rngs, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(model, grads)  # In-place updates.

@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch):
  loss, logits = loss_fn(model, rngs, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
import jax

metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}

rngs = nnx.Rngs(0)

print("Starting Training...\n")

total_start_time = time.time()
epoch_start_time = time.time()

for step, batch in enumerate(train_ds.as_numpy_iterator()):

  model.train()
  train_step(model, optimizer, metrics, rngs, batch)

  # Ensure GPU finishes this step before timing
  jax.block_until_ready(model)

  if step > 0 and (step % eval_every == 0 or step == train_steps - 1):

    # ---- Epoch timing ----
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    # Log training metrics
    for metric, value in metrics.compute().items():
      metrics_history[f'train_{metric}'].append(float(value))
    metrics.reset()

    # ---- Evaluation ----
    model.eval()
    for test_batch in test_ds.as_numpy_iterator():
      eval_step(model, metrics, rngs, test_batch)

    jax.block_until_ready(model)  # ensure eval finishes

    for metric, value in metrics.compute().items():
      metrics_history[f'test_{metric}'].append(float(value))
    metrics.reset()

    print(f"Step {step}")
    print(f"Epoch Time: {epoch_time:.4f} seconds")
    print(f"Average Time per Step: {epoch_time/eval_every:.6f} seconds\n")

    epoch_start_time = time.time()

    clear_output(wait=True)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')

    for dataset in ('train', 'test'):
      ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
      ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')

    ax1.legend()
    ax2.legend()
    plt.show()

  if step >= train_steps:
    break

# ---- Total timing ----
jax.block_until_ready(model)
total_end_time = time.time()

print("\n==============================")
print("Total Training Time:", total_end_time - total_start_time, "seconds")
print("Average Time Per Step:", (total_end_time - total_start_time)/train_steps, "seconds")
print("==============================")


model.eval() # Switch to evaluation mode.

@nnx.jit
def pred_step(model: CNN, batch):
  logits = model(batch['image'], None)
  return logits.argmax(axis=1)

test_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(model, test_batch)

fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
  ax.imshow(test_batch['image'][i, ..., 0], cmap='gray')
  ax.set_title(f'label={pred[i]}')
  ax.axis('off')

sig = [tf.TensorSpec(shape=(1, 28, 28, 1), dtype=tf.float32)]

import jax
import jax.numpy as jnp
import time
import numpy as np

# Ensure evaluation mode
model.eval()

correct = 0
total = 0

for batch in test_ds.as_numpy_iterator():
    logits = model(batch['image'], None)
    preds = jnp.argmax(logits, axis=1)

    correct += (preds == batch['label']).sum()
    total += len(batch['label'])

test_accuracy = float(correct) / total

print("=================================")
print("Test Accuracy:", test_accuracy * 100, "%")
print("=================================")

# JIT compile inference
@nnx.jit
def inference_step(model, batch):
    logits = model(batch['image'], None)
    return jnp.argmax(logits, axis=1)

# Get one test batch
test_batch = next(test_ds.as_numpy_iterator())

# Move batch to device before timing
images = jax.device_put(test_batch['image']) #step4

# -------------------------
# Warmup (JIT compilation)
# -------------------------
print("Running JIT warmup...")
pred = inference_step(model, {'image': images}) #step 2
jax.block_until_ready(pred)

# -------------------------
# Benchmark
# -------------------------

num_runs = 100

start_time = time.time()

for _ in range(num_runs):
    pred = inference_step(model, {'image': images}) #step1
    jax.block_until_ready(pred) # step 3

end_time = time.time()

total_time = end_time - start_time
avg_time = total_time / num_runs

print("\n==============================")
print("Inference Benchmark Results")
print("==============================")

print(f"Batch Size: {images.shape[0]}")
print(f"Total Runs: {num_runs}")
print(f"Total Time: {total_time:.6f} seconds")
print(f"Average Time per Batch: {avg_time:.6f} seconds")

print(f"Average Time per Image: {avg_time/images.shape[0]:.8f} seconds")
print("==============================")

start = time.time()

for batch in test_ds.as_numpy_iterator():
    pred = inference_step(model, batch)
    jax.block_until_ready(pred)

end = time.time()

print("\nFull Test Set Inference Time:", end-start, "seconds")

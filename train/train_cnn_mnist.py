
from src.models.cnn import CNN
from src.input_pipeline.inputcnn import train_ds, test_ds

import jax
import jax.numpy as jnp
import optax
import time

from flax.training import train_state


# -----------------------------
# Create Model
# -----------------------------

model = CNN()

rng = jax.random.PRNGKey(0)

sample_batch = next(train_ds.as_numpy_iterator())
sample_images = sample_batch["image"]

params = model.init(rng, sample_images)


# -----------------------------
# Optimizer
# -----------------------------

learning_rate = 0.001
tx = optax.adam(learning_rate)

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)


# -----------------------------
# Loss Function
# -----------------------------

def compute_loss(params, images, labels):

    logits = model.apply(params, images)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=labels
    ).mean()

    return loss, logits


# -----------------------------
# Training Step
# -----------------------------

@jax.jit
def train_step(state, batch):

    def loss_fn(params):

        loss, logits = compute_loss(
            params,
            batch["image"],
            batch["label"]
        )

        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (loss, logits), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    accuracy = jnp.mean(
        jnp.argmax(logits, axis=1) == batch["label"]
    )

    return state, loss, accuracy


# -----------------------------
# Evaluation Step
# -----------------------------

@jax.jit
def eval_step(state, batch):

    logits = model.apply(state.params, batch["image"])

    accuracy = jnp.mean(
        jnp.argmax(logits, axis=1) == batch["label"]
    )

    return accuracy


# -----------------------------
# Training Loop
# -----------------------------

train_steps = 1200
eval_every = 200

print("Starting Training...\n")

start_time = time.time()

train_iterator = train_ds.as_numpy_iterator()

for step in range(train_steps):

    batch = next(train_iterator)

    state, loss, acc = train_step(state, batch)

    if step % eval_every == 0:

        print(f"Step {step}")
        print(f"Loss: {float(loss):.4f}")
        print(f"Train Accuracy: {float(acc)*100:.2f}%")

        test_accs = []

        for test_batch in test_ds.as_numpy_iterator():

            acc = eval_step(state, test_batch)
            test_accs.append(acc)

        test_acc = jnp.mean(jnp.array(test_accs))

        print(f"Test Accuracy: {float(test_acc)*100:.2f}%\n")


end_time = time.time()

print("================================")
print("Total Training Time:", end_time - start_time)
print("================================")
import pickle

with open("cnn_mnist_params.pkl", "wb") as f:
    pickle.dump(state.params, f)

print("Model parameters saved!")

# src/train/train_vit_cifar10.py

import os

#  CRITICAL: prevent JAX from grabbing all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

# silence TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle

from flax.training import train_state

import tensorflow as tf
import tensorflow_datasets as tfds

#  Force TensorFlow to CPU (avoid GPU conflict with JAX)
tf.config.set_visible_devices([], 'GPU')

from src.models.vit import ViT


print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())


# -----------------------------
# Train State
# -----------------------------
class CustomTrainState(train_state.TrainState):
    dropout_rng: jax.random.PRNGKey


def create_train_state(model, params, rng, lr=3e-4):
    tx = optax.adam(lr)
    return CustomTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_rng=rng
    )


# -----------------------------
# Train Step
# -----------------------------
@jax.jit
def train_step(state, images, labels):

    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            images,
            train=True,
            rngs={"dropout": dropout_rng}
        )

        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True
    )(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)

    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    return state, loss, acc


# -----------------------------
# Eval Step
# -----------------------------
@jax.jit
def eval_step(state, images, labels):

    logits = state.apply_fn(
        {"params": state.params},
        images,
        train=False
    )

    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    return loss, acc


# -----------------------------
# STREAMING CIFAR-10 DATA
# -----------------------------
def get_cifar10_datasets(batch_size=8):  #  smaller batch for stability

    ds_train = tfds.load("cifar10", split="train[:5%]", as_supervised=True)
    ds_test = tfds.load("cifar10", split="test[:5%]", as_supervised=True)

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds_train = (
        ds_train
        .map(preprocess)
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    ds_test = (
        ds_test
        .map(preprocess)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds_train, ds_test


# -----------------------------
# Training Loop
# -----------------------------
def train_model(state, ds_train, ds_test, epochs=1):

    for epoch in range(epochs):

        train_losses = []
        train_accs = []

        for batch in tfds.as_numpy(ds_train):
            images, labels = batch

            state, loss, acc = train_step(
                state,
                jnp.array(images),
                jnp.array(labels)
            )

            train_losses.append(float(loss))
            train_accs.append(float(acc))

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)

        val_losses = []
        val_accs = []

        for batch in tfds.as_numpy(ds_test):
            images, labels = batch

            loss, acc = eval_step(
                state,
                jnp.array(images),
                jnp.array(labels)
            )

            val_losses.append(float(loss))
            val_accs.append(float(acc))

        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    return state


# -----------------------------
# Main
# -----------------------------
def main():

    print("Loading CIFAR-10 (streaming)...")

    ds_train, ds_test = get_cifar10_datasets(batch_size=8)

    # SMALL ViT (fits your GPU)
    model = ViT(
        patch_size=4,
        embed_dim=32,
        hidden_dim=64,
        n_heads=2,
        mlp_dim=64,
        num_layers=2,
        drop_p=0.1,
        num_classes=10
    )

    rng = jax.random.PRNGKey(0)

    dummy = jnp.ones((1, 32, 32, 3))

    # FIXED PARAMS BUG HERE
    params = model.init(rng, dummy)["params"]

    state = create_train_state(model, params, rng)

    print("\nStarting training...\n")

    state = train_model(state, ds_train, ds_test, epochs=1)

    # Save model
    with open("vit_cifar10_params.pkl", "wb") as f:
        pickle.dump(state.params, f)

    print("\nViT CIFAR-10 parameters saved!")


if __name__ == "__main__":
    main()
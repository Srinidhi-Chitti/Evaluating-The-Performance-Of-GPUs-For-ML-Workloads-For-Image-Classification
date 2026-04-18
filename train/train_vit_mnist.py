# src/train/train_vit_mnist.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF logs

import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax.core.frozen_dict import freeze
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')

from src.models.vit import ViT  # Make sure src/models/vit.py exists


# -----------------------------
# Check JAX devices
# -----------------------------
print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

# -----------------------------
# Custom TrainState to track dropout RNG
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
# Training and evaluation steps
# -----------------------------
@jax.jit
def train_step(state, batch):
    images, labels = batch
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            images,
            train=True,
            rngs={'dropout': dropout_rng}
        )
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return state, loss, acc

@jax.jit
def eval_step(state, batch):
    images, labels = batch
    logits = state.apply_fn({'params': state.params}, images, train=False)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, acc

# -----------------------------
# Data loading
# -----------------------------
def load_mnist_data():
    ds_train = tfds.load("mnist", split="train", as_supervised=True, shuffle_files=True)
    ds_test = tfds.load("mnist", split="test", as_supervised=True)

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, (28, 28, 1))
        return image, label

    ds_train = ds_train.map(preprocess).cache().shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

    x_train, y_train = [], []
    for batch in tfds.as_numpy(ds_train):
        images, labels = batch
        x_train.append(images)
        y_train.append(labels)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test, y_test = [], []
    for batch in tfds.as_numpy(ds_test):
        images, labels = batch
        x_test.append(images)
        y_test.append(labels)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    return (x_train, y_train), (x_test, y_test)

# -----------------------------
# Training loop
# -----------------------------
def train_model(state, x_train, y_train, x_val, y_val, batch_size=128, epochs=10):
    num_batches = len(x_train) // batch_size
    for epoch in range(epochs):
        # Shuffle
        perms = np.random.permutation(len(x_train))
        epoch_loss = []
        epoch_acc = []
        for i in range(num_batches):
            idx = perms[i*batch_size:(i+1)*batch_size]
            batch_imgs = jnp.array(x_train[idx])
            batch_labels = jnp.array(y_train[idx])
            state, loss, acc = train_step(state, (batch_imgs, batch_labels))
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))

        train_loss = np.mean(epoch_loss)
        train_acc = np.mean(epoch_acc)

        val_loss, val_acc = eval_step(state, (jnp.array(x_val), jnp.array(y_val)))

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {float(val_loss):.4f} | Val Acc: {float(val_acc):.4f}")

    return state

# -----------------------------
# Main
# -----------------------------
def main():
    print("Initializing model...")
    model = ViT(
        patch_size=7,
        embed_dim=64,
        hidden_dim=128,
        n_heads=4,
        mlp_dim=128,
        num_layers=4,
        drop_p=0.1,
        num_classes=10
    )

    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 28, 28, 1))
    params = model.init(rng, dummy)

    state = create_train_state(model, params['params'], rng)

    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print("Data loaded:", x_train.shape, x_test.shape)

    print("Starting training...\n")
    state = train_model(state, x_train, y_train, x_test, y_test, batch_size=128, epochs=10)

    # Save trained parameters
    import pickle
    with open("vit_mnist_params.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print("ViT MNIST model parameters saved!")

if __name__ == "__main__":
    main()
# src/benchmarkdata/vit_benchmark.py

import time
import pickle
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf

from src.models.vit import ViT

# ----------------------------------
# SYSTEM SETUP
# ----------------------------------
tf.config.set_visible_devices([], 'GPU')  # avoid TF GPU memory conflict

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())


# ----------------------------------
# DATA LOADERS (MEMORY SAFE)
# ----------------------------------
def load_mnist(batch_size=16):
    ds = tfds.load("mnist", split="test[:10%]", as_supervised=True)

    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    ds = ds.map(preprocess).batch(batch_size).take(20)
    return tfds.as_numpy(ds)


def load_cifar10(batch_size=16):
    ds = tfds.load("cifar10", split="test[:5%]", as_supervised=True)

    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    ds = ds.map(preprocess).batch(batch_size).take(20)
    return tfds.as_numpy(ds)


# ----------------------------------
# EVALUATION (JAX CORRECT)
# ----------------------------------
def evaluate(model, params, dataset):

    total = 0
    correct = 0

    @jax.jit
    def forward(params, images):
        return model.apply({"params": params}, images, train=False)

    # -------- Warmup --------
    for images, labels in dataset:
        images = jax.device_put(jnp.array(images))
        logits = forward(params, images)
        logits.block_until_ready()
        break

    # -------- Timing --------
    start = time.time()

    for images, labels in dataset:
        images = jax.device_put(jnp.array(images))

        logits = forward(params, images)
        logits.block_until_ready()

        preds = jnp.argmax(logits, axis=-1)

        correct += int(jnp.sum(preds == labels))
        total += len(labels)

    end = time.time()

    accuracy = correct / total
    latency = end - start

    return accuracy, latency


# ----------------------------------
# MNIST BENCHMARK
# ----------------------------------
def benchmark_mnist():

    print("\n--- ViT MNIST ---")

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

    dummy = jnp.ones((1, 28, 28, 1))
    params = model.init(jax.random.PRNGKey(0), dummy)["params"]

    with open("vit_mnist_params.pkl", "rb") as f:
        params = pickle.load(f)

    ds = load_mnist()

    acc, latency = evaluate(model, params, ds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Inference Time: {latency:.2f} sec")


# ----------------------------------
# CIFAR-10 BENCHMARK
# ----------------------------------
def benchmark_cifar10():

    print("\n--- ViT CIFAR-10 ---")

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

    dummy = jnp.ones((1, 32, 32, 3))
    params = model.init(jax.random.PRNGKey(0), dummy)["params"]

    with open("vit_cifar10_params.pkl", "rb") as f:
        params = pickle.load(f)

    ds = load_cifar10()

    acc, latency = evaluate(model, params, ds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Inference Time: {latency:.2f} sec")


# ----------------------------------
# MAIN
# ----------------------------------
def main():
    benchmark_mnist()
    benchmark_cifar10()


if __name__ == "__main__":
    main()
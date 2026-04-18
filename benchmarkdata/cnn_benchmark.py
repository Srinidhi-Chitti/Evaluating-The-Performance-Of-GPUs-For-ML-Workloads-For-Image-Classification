import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import time
import jax
import jax.numpy as jnp
import numpy as np
import pickle

from src.models.cnn import initialize_model
from src.data.dataloader import load_mnist, load_cifar10


print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())


# --------------------------------------------------
# Initialize Model (global for JIT)
# --------------------------------------------------

model, _ = initialize_model()


# --------------------------------------------------
# JIT inference function
# --------------------------------------------------

@jax.jit
def inference_step(params, images):

    logits = model.apply(params, images)

    return jnp.argmax(logits, axis=1)


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

def compute_accuracy(params, images, labels):

    batch_size = 128
    correct = 0
    total = 0

    for i in range(0, len(images), batch_size):

        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        batch_images = jnp.array(batch_images)

        preds = inference_step(params, batch_images)

        preds.block_until_ready()

        correct += np.sum(np.array(preds) == batch_labels)
        total += len(batch_labels)

    return correct / total


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

def benchmark_inference(params, images):

    batch = jnp.array(images[:128])

    batch = jax.device_put(batch)

    print("\nRunning JIT warmup...")

    preds = inference_step(params, batch)
    preds.block_until_ready()

    runs = 100

    start = time.time()

    for _ in range(runs):

        preds = inference_step(params, batch)
        preds.block_until_ready()

    end = time.time()

    total_time = end - start
    avg_batch = total_time / runs
    avg_image = avg_batch / batch.shape[0]

    print("\nInference Benchmark Results")
    print("===========================")

    print("Batch size:", batch.shape[0])
    print("Runs:", runs)
    print("Total time:", total_time, "seconds")
    print("Average batch time:", avg_batch, "seconds")
    print("Average image time:", avg_image, "seconds")


# --------------------------------------------------
# Run Benchmark per Dataset
# --------------------------------------------------

def run_dataset(dataset_name):

    print("\n===================================")
    print("Dataset:", dataset_name)
    print("===================================")

    if dataset_name == "mnist":

        with open("cnn_mnist_params.pkl", "rb") as f:
            params = pickle.load(f)

        (x_train, y_train), (x_test, y_test) = load_mnist()

    elif dataset_name == "cifar10":

        with open("cnn_cifar10_params.pkl", "rb") as f:
            params = pickle.load(f)

        (x_train, y_train), (x_test, y_test) = load_cifar10()

    x_test = x_test.astype(np.float32)

    accuracy = compute_accuracy(params, x_test, y_test)

    print("\nAccuracy:", accuracy * 100, "%")

    benchmark_inference(params, x_test)


# --------------------------------------------------
# Run BOTH datasets
# --------------------------------------------------

run_dataset("mnist")
run_dataset("cifar10")
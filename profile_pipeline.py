import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 🔥 suppress warnings

import pickle
import jax
import jax.numpy as jnp
import jax.profiler
import tensorflow_datasets as tfds
import tensorflow as tf
from jax import lax
import time
from functools import partial

from src.models.cnn import CNN

# ------------------------------
# SYSTEM SETUP
# ------------------------------
tf.config.set_visible_devices([], "GPU")

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

# ------------------------------
# KERNELS
# ------------------------------
BLUR_KERNEL = jnp.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=jnp.float32) / 16.0

SOBEL_X = jnp.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=jnp.float32)


def prepare_kernels():
    blur = BLUR_KERNEL[:, :, None, None]
    sobel = SOBEL_X[:, :, None, None]

    blur = jnp.repeat(blur, 3, axis=3)
    sobel = jnp.repeat(sobel, 3, axis=3)

    return blur, sobel


BLUR_K, SOBEL_K = prepare_kernels()

# ------------------------------
# MODEL
# ------------------------------
model = CNN(num_classes=10)

dummy = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
_ = model.init(jax.random.PRNGKey(0), dummy)

with open("cnn_cifar10_params.pkl", "rb") as f:
    loaded = pickle.load(f)

variables = loaded if "params" in loaded else {"params": loaded}

# ------------------------------
# DATA
# ------------------------------
def load_cifar10(batch_size):
    ds = tfds.load("cifar10", split="test[:2%]", as_supervised=True)

    def preprocess(x, y):
        x = tf.cast(x, tf.float32)
        return x, y

    ds = ds.map(preprocess)
    ds = ds.batch(batch_size)

    return tfds.as_numpy(ds)

# ------------------------------
# PIPELINE
# ------------------------------
@partial(jax.jit, donate_argnums=(1,))
def full_pipeline(variables, images):

    x = images * (1.0 / 255.0)

    x = lax.conv_general_dilated(
        x, BLUR_K, (1, 1), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=3
    )

    x = lax.conv_general_dilated(
        x, SOBEL_K, (1, 1), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=3
    )

    logits = model.apply(variables, x)
    return jax.nn.softmax(logits)

# ------------------------------
# MAIN
# ------------------------------
def run():
    print("\n--- RUNNING PIPELINE ---")

    batch_sizes = [32, 64]

    for bs in batch_sizes:
        print(f"\n🔹 Batch Size: {bs}")

        dataset = load_cifar10(batch_size=bs)

        # --------------------------
        # WARMUP
        # --------------------------
        print("Warming up...")

        for i, (images, _) in enumerate(dataset):
            images = jax.device_put(images)

            for _ in range(2):
                full_pipeline(variables, images).block_until_ready()

            if i == 2:
                break

        time.sleep(1)

        # --------------------------
        # TIMING
        # --------------------------
        times = []

        print("Running...")

        for run_id in range(2):
            start = time.perf_counter()

            for images, _ in dataset:
                images = jax.device_put(images)
                preds = full_pipeline(variables, images)
                preds.block_until_ready()

            end = time.perf_counter()

            elapsed = end - start
            times.append(elapsed)

            print(f"Run {run_id+1}: {elapsed:.4f} sec")

        avg = sum(times) / len(times)
        print(f"✅ Avg Time: {avg:.4f} sec")

        # --------------------------
        # PROFILING
        # --------------------------
        trace_dir = f"/tmp/profile-bs-{bs}"

        if os.path.exists(trace_dir):
            os.system(f"rm -rf {trace_dir}")

        print("Profiling...")

        jax.profiler.start_trace(trace_dir)

        for images, _ in dataset:
            images = jax.device_put(images)
            preds = full_pipeline(variables, images)
            preds.block_until_ready()

        jax.profiler.stop_trace()

        print(f"✅ Profile saved at {trace_dir}")
        print(f"👉 Run: xprof --port 88{bs} {trace_dir}")

# ------------------------------
if __name__ == "__main__":
    run()
ssss

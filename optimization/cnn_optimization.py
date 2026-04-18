# src/optimization/cnn_optimization.py

import pickle
import jax
import jax.numpy as jnp
import jax.profiler
import tensorflow_datasets as tfds
import tensorflow as tf

from src.models.cnn import CNN

# ------------------------------
# SYSTEM SETUP
# ------------------------------
tf.config.set_visible_devices([], "GPU")
print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

# ------------------------------
# DATA LOADER
# ------------------------------
def load_cifar10(batch_size=64):  # increased batch size
    ds = tfds.load("cifar10", split="test[:5%]", as_supervised=True)

    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return tfds.as_numpy(ds)

# ------------------------------
# MAIN FUNCTION
# ------------------------------
def run():
    print("\n--- CNN Optimization Run ---")

    # ------------------------------
    # MODEL INIT
    # ------------------------------
    model = CNN(num_classes=10)
    dummy = jnp.ones((1, 32, 32, 3))
    variables = model.init(jax.random.PRNGKey(0), dummy)

    # Load trained params
    with open("cnn_cifar10_params.pkl", "rb") as f:
        loaded = pickle.load(f)

    variables = loaded if isinstance(loaded, dict) and "params" in loaded else {"params": loaded}

    # ------------------------------
    # LOAD DATA
    # ------------------------------
    dataset = load_cifar10()

    # Move to device ONCE (important)
    dataset = [(jax.device_put(images), labels) for images, labels in dataset]

    # ------------------------------
    # JIT INFERENCE (FUSED)
    # ------------------------------
    @jax.jit
    def inference_step(variables, images):
        logits = model.apply(variables, images)
        return jax.nn.softmax(logits)

    # ------------------------------
    # JAXPR INSPECTION (ADD HERE)
    # ------------------------------
    print("\n🔬 JAXPR OUTPUT:")
    sample_images = jnp.ones((64, 32, 32, 3))
    print(jax.make_jaxpr(inference_step)(variables, sample_images))

    # ------------------------------
    # WARMUP
    # ------------------------------
    print("\n Warming up...")
    for i, (images, _) in enumerate(dataset):
        inference_step(variables, images).block_until_ready()
        if i == 2:
            break

    # ------------------------------
    # PROFILING
    # ------------------------------
    jax.profiler.start_trace("/tmp/profile-data")
    print(" Profiling started...")

    for images, _ in dataset:
        with jax.profiler.TraceAnnotation("cnn_inference"):
            preds = inference_step(variables, images)
            preds.block_until_ready()

    jax.profiler.stop_trace()
    print(" Trace saved")

    # MEMORY PROFILE

    print(" Saving memory profile...")
    for images, _ in dataset:
        preds = inference_step(variables, images)
        preds.block_until_ready()
        break
    jax.profiler.save_device_memory_profile("cnn_memory.prof")

    print("\n Done!")
    print(" Run: xprof --port 8791 /tmp/profile-data")
# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    run()
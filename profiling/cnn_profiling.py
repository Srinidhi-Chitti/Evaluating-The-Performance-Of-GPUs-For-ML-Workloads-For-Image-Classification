# src/profiling/cnn_profiling.py

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
tf.config.set_visible_devices([], "GPU")  # Avoid TF grabbing GPU
print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

# ------------------------------
# DATA LOADER
# ------------------------------
def load_cifar10(batch_size=4):  # small batch for profiling
    ds = tfds.load("cifar10", split="test[:5%]", as_supervised=True)

    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    ds = ds.take(5).map(preprocess).batch(batch_size)  # fewer batches
    return tfds.as_numpy(ds)

# ------------------------------
# MAIN PROFILING FUNCTION
# ------------------------------
def profile_cnn():
    print("\n--- Profiling CNN CIFAR-10 (Optimized) ---")

    # Initialize model
    model = CNN(num_classes=10)
    dummy = jnp.ones((1, 32, 32, 3))
    _ = model.init(jax.random.PRNGKey(0), dummy)

    # Load trained params
    with open("cnn_cifar10_params.pkl", "rb") as f:
        loaded = pickle.load(f)

    # Fix variable structure if needed
    variables = loaded if isinstance(loaded, dict) and "params" in loaded else {"params": loaded}
    print("✔ Variables structure:", variables.keys())

    # Load data
    dataset = load_cifar10()

    # JIT compiled forward function
    @jax.jit
    def forward(variables, images):
        return model.apply(variables, images)

    # ------------------------------
    # WARMUP
    # ------------------------------
    for images, _ in dataset:
        images = jnp.array(images)  # avoid device_put until needed
        logits = forward(variables, images)
        logits.block_until_ready()
        break

    # ------------------------------
    # XPROF TRACE
    # ------------------------------
    jax.profiler.start_trace("/tmp/profile-data")
    print(" Profiling started...")

    for images, _ in dataset:
        with jax.profiler.TraceAnnotation("cnn_inference"):
            images = jnp.array(images)
            logits = forward(variables, images)
            logits.block_until_ready()

    jax.profiler.stop_trace()
    print("✅ XProf trace saved to /tmp/profile-data")

    # ------------------------------
    # DEVICE MEMORY PROFILE (SAFE)
    # ------------------------------
    print("💾 Saving device memory profile...")
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (500, 500)) 

    def func1(x):
        return jnp.tile(x, 5) * 0.5  # reduce size

    def func2(x):
        y = func1(x)
        return y, jnp.tile(x, 5) + 1

    y, z = func2(x)
    z.block_until_ready()
    jax.profiler.save_device_memory_profile("memory.prof")
    print("✅ Device memory profile saved as memory.prof")

    print("Profiling complete!")
    print("View XProf trace: xprof --port 8791 /tmp/profile-data")
    print(" Memory profile file: memory.prof")

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    profile_cnn()
import time
import os
import pickle
import jax
import jax.numpy as jnp
import jax.profiler
import tensorflow_datasets as tfds
import tensorflow as tf

from src.models.vit import ViT

# جلوگیری از GPU conflict (TensorFlow off GPU)
tf.config.set_visible_devices([], 'GPU')

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())


# ----------------------------
# MNIST LOADER (FIXED)
# ----------------------------
def load_mnist(batch_size=16):
    ds = tfds.load("mnist", split="test[:5%]", as_supervised=True)

    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        # ✅ KEEP ORIGINAL: 28x28x1 (DO NOT convert to 3 channels)
        return x, y

    ds = ds.map(preprocess).batch(batch_size).take(20)
    return tfds.as_numpy(ds)


def main():
    print("\n--- Profiling ViT on MNIST ---")

    # ----------------------------
    # ✅ MODEL (MATCH PARAMS)
    # ----------------------------
    model = ViT(
        patch_size=7,      # 🔥 IMPORTANT (from error)
        embed_dim=64,      # 🔥 IMPORTANT
        hidden_dim=128,
        n_heads=4,
        mlp_dim=128,
        num_layers=2,
        drop_p=0.1,
        num_classes=10
    )

    # ✅ MATCH INPUT SHAPE
    dummy = jnp.ones((1, 28, 28, 1))
    params = model.init(jax.random.PRNGKey(0), dummy)["params"]

    # ----------------------------
    # LOAD TRAINED WEIGHTS
    # ----------------------------
    with open("vit_mnist_params.pkl", "rb") as f:
        params = pickle.load(f)

    dataset = load_mnist()

    @jax.jit
    def forward(params, images):
        return model.apply({"params": params}, images, train=False)

    # ----------------------------
    # 🔥 STRONG WARMUP (FIX TIMER ERROR)
    # ----------------------------
    print("Warming up...")
    for i, (images, _) in enumerate(dataset):
        images = jax.device_put(jnp.array(images))

        for _ in range(3):
            forward(params, images).block_until_ready()

        if i == 3:
            break

    time.sleep(1)

    # ----------------------------
    # PROFILING
    # ----------------------------
    trace_dir = "/tmp/profile-vit-mnist"

    if os.path.exists(trace_dir):
        os.system(f"rm -rf {trace_dir}")

    jax.profiler.start_trace(trace_dir)
    print("Profiling started...")

    start = time.time()

    for images, labels in dataset:
        with jax.profiler.TraceAnnotation("vit_inference"):

            images = jax.device_put(jnp.array(images))

            logits = forward(params, images)

            logits.block_until_ready()

    end = time.time()

    jax.profiler.stop_trace()

    print("✅ Done (MNIST)")
    print(f"Inference Time: {end - start:.2f} sec")
    print("👉 Run: xprof --port 8720 /tmp/profile-vit-mnist")

    # Optional memory profile
    jax.profiler.save_device_memory_profile("vit_mnist_memory.prof")


if __name__ == "__main__":
    main()



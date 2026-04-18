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
# CIFAR-10 LOADER
# ----------------------------
def load_cifar10(batch_size=16):
    ds = tfds.load("cifar10", split="test[:5%]", as_supervised=True)

    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    ds = ds.map(preprocess).batch(batch_size).take(20)
    return tfds.as_numpy(ds)


def main():
    print("\n--- Profiling ViT on CIFAR-10 ---")

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

    # Load trained weights
    with open("vit_cifar10_params.pkl", "rb") as f:
        params = pickle.load(f)

    dataset = load_cifar10()

    @jax.jit
    def forward(params, images):
        return model.apply({"params": params}, images, train=False)

    # ----------------------------
    # 🔥 WARMUP (IMPORTANT)
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
    trace_dir = "/tmp/profile-vit-cifar"

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

    print("Done (CIFAR-10) loading.The below is the inference time")
    print(f"Inference Time: {end - start:.2f} sec")
    print("Run: xprof --port random number tmp/profile-vit-cifar")


if __name__ == "__main__":
    main()

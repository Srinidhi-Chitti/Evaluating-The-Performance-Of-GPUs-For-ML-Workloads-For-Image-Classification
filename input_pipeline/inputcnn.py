import tensorflow_datasets as tfds
import tensorflow as tf

tf.random.set_seed(0)

train_steps = 1200
eval_every = 200
batch_size = 32

train_ds = tfds.load('mnist', split='train')
test_ds = tfds.load('mnist', split='test')

train_ds = train_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)

test_ds = test_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)

train_ds = train_ds.repeat().shuffle(1024)
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)

test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

def preprocess_cifar10(image, label):

    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)

    return {
        "image": image,
        "label": label
    }


batch_size = 128


# Load datasets
train_cifar10 = tfds.load("cifar10", split="train", as_supervised=True)
test_cifar10 = tfds.load("cifar10", split="test", as_supervised=True)


# Build pipelines
train_ds_cifar10 = (
    train_cifar10
    .map(preprocess_cifar10)
    .shuffle(10000)
    .repeat()        
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds_cifar10 = (
    test_cifar10
    .map(preprocess_cifar10)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
import tensorflow as tf
import numpy as np
import os

print(tf.__version__)
#https://www.tensorflow.org/tutorials/distribute/custom_training

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Add a dimension to the array -> new shape == (28, 28, 1)
# This is done because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
# batch_size dimension will be added later on.
train_images = train_images[..., None]
test_images = test_images[..., None]

# Scale the images to the [0, 1] range.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)


strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


def create_model():
  regularizer = tf.keras.regularizers.L2(1e-5)
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3,
                             activation='relu',
                             kernel_regularizer=regularizer),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3,
                             activation='relu',
                             kernel_regularizer=regularizer),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64,
                            activation='relu',
                            kernel_regularizer=regularizer),
      tf.keras.layers.Dense(10, kernel_regularizer=regularizer)
    ])

  return model

# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


with strategy.scope():
  # Set reduction to `NONE` so you can do the reduction yourself.
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True,
      reduction=tf.keras.losses.Reduction.NONE)
  def compute_loss(labels, predictions, model_losses):
    per_example_loss = loss_object(labels, predictions)
    loss = tf.nn.compute_average_loss(per_example_loss)
    if model_losses:
      loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
    return loss

with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')


with strategy.scope():
  model = create_model()

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


def train_step(inputs):
  images, labels = inputs

  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = compute_loss(labels, predictions, model.losses)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_accuracy.update_state(labels, predictions)
  return loss

def test_step(inputs):
  images, labels = inputs

  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss.update_state(t_loss)
  test_accuracy.update_state(labels, predictions)


# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
  return strategy.run(test_step, args=(dataset_inputs,))

for epoch in range(EPOCHS):
  # TRAIN LOOP
  total_loss = 0.0
  num_batches = 0
  for x in train_dist_dataset:
    total_loss += distributed_train_step(x)
    num_batches += 1
  train_loss = total_loss / num_batches

  # TEST LOOP
  for x in test_dist_dataset:
    distributed_test_step(x)

  if epoch % 2 == 0:
    checkpoint.save(checkpoint_prefix)

  template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
              "Test Accuracy: {}")
  print(template.format(epoch + 1, train_loss,
                         train_accuracy.result() * 100, test_loss.result(),
                         test_accuracy.result() * 100))

  test_loss.reset_states()
  train_accuracy.reset_states()
  test_accuracy.reset_states()


eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='eval_accuracy')

new_model = create_model()
new_optimizer = tf.keras.optimizers.Adam()

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)



@tf.function
def eval_step(images, labels):
  predictions = new_model(images, training=False)
  eval_accuracy(labels, predictions)


checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, model=new_model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for images, labels in test_dataset:
  eval_step(images, labels)

print('Accuracy after restoring the saved model without strategy: {}'.format(
    eval_accuracy.result() * 100))


import os
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt


import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
import time

tdfs_list_num = 414
IMG_SIZE = 224
LABELS = ["cat", "dog"]

print('terraform version:', tf.__version__)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
tfds.disable_progress_bar()
print(str(tfds.list_builders()[:tdfs_list_num]))

(train_ds, validation_ds), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True
)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


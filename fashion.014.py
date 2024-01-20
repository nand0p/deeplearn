from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os

NUM_EPOCHS = 200
BATCH_SIZE = 128
INIT_LR = 2e-4

((trainX, _), (testX, _)) = fashion_mnist.load_data()
trainImages = np.concatenate([trainX, testX])
trainImages = np.expand_dims(trainImages, axis=-1)
trainImages = (trainImages.astype("float") - 127.5) / 127.5

class GANModel:
  def build_generator(dim, depth, channels=1, inputDim=100, outputDim=512):
    model = Sequential()
    inputShape = (dim, dim, depth)
    chanDim = -1

    model.add(layers.Dense(input_dim=inputDim, units=outputDim))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(dim * dim * depth))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization())

    model.add(layers.Reshape(inputShape))
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization(axis=chanDim))

    model.add(layers.Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.Activation("tanh"))
    return model

  def build_discriminator(width, height, depth, alpha=0.2):
    model = Sequential()
    inputShape = (height, width, depth)

    model.add(layers.Conv2D(32, (5, 5), padding="same", strides=(2, 2), input_shape=inputShape))
    model.add(layers.LeakyReLU(alpha=alpha))

    model.add(layers.Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Flatten())

    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=alpha))

    model.add(layers.Dense(1))
    model.add(layers.Activation("sigmoid"))
    return model

gen = GANModel.build_generator(7, 64, channels=1)
gen.summary()
disc = GANModel.build_discriminator(28, 28, 1)
disc.summary()
discOpt = Adam(learning_rate=INIT_LR, beta_1=0.5)
disc.compile(loss="binary_crossentropy", optimizer=discOpt)

disc.trainable = False
ganInput = Input(shape=(100,))
ganOutput = disc(gen(ganInput))
gan = Model(ganInput, ganOutput)
ganOpt = Adam(learning_rate=INIT_LR, beta_1=0.5)
gan.compile(loss="binary_crossentropy", optimizer=discOpt)

benchmarkNoise = np.random.uniform(-1, 1, size=(256, 100))

for epoch in range(0, NUM_EPOCHS):
  print("[INFO] starting epoch {} of {}...".format(epoch + 1, NUM_EPOCHS))
  batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE)
  # loop over the batches
  for i in range(0, batchesPerEpoch):
    # initialize an (empty) output path
    p = None
    # select the next batch of images, then randomly generate
    # noise for the generator to predict on
    imageBatch = trainImages[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
    noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

    # generate images using the noise + generator model
    genImages = gen.predict(noise, verbose=0)
    # concatenate the *actual* images and the *generated* images,
    # construct class labels for the discriminator, and shuffle
    # the data
    X = np.concatenate((imageBatch, genImages))
    y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
    y = np.reshape(y, (-1,))
    (X, y) = shuffle(X, y)
    # train the discriminator on the data
    discLoss = disc.train_on_batch(X, y)

    # let's now train our generator via the adversarial model by
    # (1) generating random noise and (2) training the generator
    # with the discriminator weights frozen
    noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
    fakeLabels = [1] * BATCH_SIZE
    fakeLabels = np.reshape(fakeLabels, (-1,))
    #ganLoss = gan.train_on_batch(noise, fakeLabels)

    # check to see if this is the end of an epoch, and if so,
    # initialize the output path
    if i == batchesPerEpoch - 1:
      p = [args["output"], "epoch_{}_output.png".format(
        str(epoch + 1).zfill(4))]
    # otherwise, check to see if we should visualize the current
    # batch for the epoch
    else:
      # create more visualizations early in the training
      # process
      if epoch < 10 and i % 25 == 0:
        p = ['/', "epoch_{}_step_{}.png".format(
          str(epoch + 1).zfill(4), str(i).zfill(5))]
      # visualizations later in the training process are less
      # interesting
      elif epoch >= 10 and i % 100 == 0:
        p = [args["output"], "epoch_{}_step_{}.png".format(
          str(epoch + 1).zfill(4), str(i).zfill(5))]

    # check to see if we should visualize the output of the
    # generator model on our benchmark data
    if p is not None:
      # show loss information
      print("[INFO] Step {}_{}: discriminator_loss={:.6f}, "
        "adversarial_loss={:.6f}".format(epoch + 1, i,
          discLoss, 0))
      # make predictions on the benchmark noise, scale it back
      # to the range [0, 255], and generate the montage
      images = gen.predict(benchmarkNoise)
      images = ((images * 127.5) + 127.5).astype("uint8")
      images = np.repeat(images, 3, axis=-1)
      vis = build_montages(images, (28, 28), (16, 16))[0]
      # write the visualization to disk
      p = os.path.sep.join(p)
      cv2.imwrite(p, vis)

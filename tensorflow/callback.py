import os
from tensorflow import keras

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


def run_training(epochs=1):
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()

    # Open a strategy scope and create/restore the model
    with strategy.scope():
        model = make_or_restore_model()

    callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        )
    ]
    model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=2,
    )


# Running the first time creates the model
run_training(epochs=1)

# Calling the same function again will resume from where we left off
run_training(epochs=1)


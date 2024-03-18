import tensorflow as tf


# base64
#plaintext = tf.constant("this is base64 encoding")
#ciphertext = tf.constant("dGhpcyBpcyBiYXNlNjQgZW5jb2Rpbmc=")


plaintext = tf.constant("this is aes")
ciphertext = tf.constant("EnCt27d536f60aa5fa38cc6e8cf3c3b0e095a29a9c0ca7d536f60aa5fa38cc6e8cf3cL1tkYFfi6AE\nXy1Se72W7NMwZOO4fUnpCaQ==IwEmS")

# Define the model
#model = tf.keras.Sequential([ tf.keras.layers.Embedding(10, 128),
#                              tf.keras.layers.LSTM(128),
#                              tf.keras.layers.Dense(10, activation="softmax") ])

model = tf.keras.Sequential([ tf.keras.layers.Dense(128, activation='relu'),
                              tf.keras.layers.Dense(64, activation='relu'),
                              tf.keras.layers.Dense(1, activation='sigmoid') ])

# Compile the model
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(plaintext, ciphertext, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(plaintext, ciphertext)

# Print the loss and accuracy
print("Loss:", loss)
print("Accuracy:", accuracy)

import tensorflow as tf        
        
#Load the dataset        
dataset = tf.keras.datasets.cifar10        
        
#Define the data preprocessor        
data_preprocessor = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)        
        
#Define the generator        
generator = data_preprocessor.flow_from_directory(        
    './data',        
    batch_size=32,        
    shuffle=True,        
    subset='train'        
)        
        
#Define the model        
model = tf.keras.models.Sequential([        
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),        
    tf.keras.layers.MaxPooling2D((2, 2)),        
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),        
    tf.keras.layers.MaxPooling2D((2, 2)),        
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),        
    tf.keras.layers.MaxPooling2D((2, 2)),        
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),        
    tf.keras.layers.MaxPooling2D((2, 2)),        
    tf.keras.layers.Flatten(),        
    tf.keras.layers.Dense(128, activation='relu'),        
    tf.keras.layers.Dense(10, activation='softmax')        
])        
        
#Define the optimizer and the loss function        
optimizer = tf.keras.optimizers.Adam()        
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)        
        
#Train the model        
model.compile(optimizer=optimizer, loss=loss_function)        
model.fit(generator, epochs=10)        
        
#Save the model        
model.save('cifar10_model.h5')        
        
#Evaluate the model        
loss, accuracy = model.evaluate(test_data)        
print(f'Test accuracy: {accuracy}')      

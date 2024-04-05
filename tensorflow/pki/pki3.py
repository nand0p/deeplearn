import tensorflow as tf        
        
#load the dataset        
dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data))        
        
#define the model        
model = tf.keras.Sequential([        
    tf.keras.layers.Embedding(input_dim=len(alphabet), output_dim=len(alphabet)),        
    tf.keras.layers.LSTM(units=len(alphabet), return_sequences=True),        
    tf.keras.layers.Dense(len(alphabet))        
])        
        
#define the loss function        
loss = tf.keras.losses.SparseCategoricalCrossentropy()        
        
#define the optimizer        
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)        
        
#train the model        
model.compile(loss=loss, optimizer=optimizer)        
        
#train the model        
model.fit(dataset, epochs=10, batch_size=32)     

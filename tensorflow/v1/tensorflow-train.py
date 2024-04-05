import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data for text classification
texts = [
  'I am happy',
  'I am sad',
  'That is a great car',
  'This movie is terrible'
]
labels = np.array([1, 0, 1, 0])

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=5, truncating='post')

model = Sequential([
  Embedding(100, 16, input_length=5),
  GlobalAveragePooling1D(),
  Dense(24, activation='relu'),
  Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=10)

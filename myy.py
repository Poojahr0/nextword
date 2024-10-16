import random

# Sample text (you can expand it)
text = "I am learning Python programming and I enjoy coding in Python"

# Tokenize text into words
words = text.lower().split()

# Create a dictionary to hold word transitions
transition_dict = {}

# Populate the transition dictionary with the next word after each word
for i in range(len(words) - 1):
    if words[i] not in transition_dict:
        transition_dict[words[i]] = []
    transition_dict[words[i]].append(words[i + 1])

# Define a function to predict the next word
def predict_next_word(word):
    word = word.lower()
    if word in transition_dict:
        return random.choice(transition_dict[word])
    else:
        return "No prediction available"

# Test the prediction
input_word = "Python"
predicted_word = predict_next_word(input_word)
print(f"Input word: {input_word}, Predicted next word: {predicted_word}")
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Sample text (you can expand it with more data)
text = "I am learning Python programming and I enjoy coding in Python"

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]

# Prepare input data for training
X = []
y = []
for i in range(1, len(sequences)):
    X.append(sequences[:i])
    y.append(sequences[i])

# Pad sequences to ensure equal length input
X = pad_sequences(X, maxlen=5, padding='pre')
y = np.array(y)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=5))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=200, verbose=1)

# Function to predict the next word
def predict_next_word(input_text):
    seq = tokenizer.texts_to_sequences([input_text])[0]
    seq = pad_sequences([seq], maxlen=5, padding='pre')
    pred = model.predict(seq, verbose=0)
    next_word = tokenizer.index_word[np.argmax(pred)]
    return next_word

# Test the next-word prediction
input_text = "I am learning"
predicted_word = predict_next_word(input_text)
print(f"Input text: '{input_text}', Predicted next word: '{predicted_word}'")

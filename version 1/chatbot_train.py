import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Load data from the JSON file
with open("medical_intents.json") as file:
    data = json.load(file)

# Extract intent information
intents = data["intents"]

# Preprocessing steps
stemmer = LancasterStemmer()
words = []
labels = []
docs_x = []
docs_y = []

for intent in intents:
    for pattern in intent["patterns"]:
        # Tokenize and stem each word in the pattern
        wrds = [stemmer.stem(word.lower()) for word in nltk.word_tokenize(pattern)]
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Build vocabulary
words = sorted(list(set(words)))

# Create bag-of-words representation for each input pattern
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = [1 if w in doc else 0 for w in words]

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# Convert to numpy arrays
training = np.array(training)
output = np.array(output)

# Define your LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(training.shape[1], 1), return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(units=64, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(units=64, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(units=64))
model.add(BatchNormalization())
model.add(Dense(units=len(labels), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.expand_dims(training, axis=-1), output, epochs=20, batch_size=32, validation_split=0.1)

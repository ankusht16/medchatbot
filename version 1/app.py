from flask import Flask, render_template, request
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random

app = Flask(__name__)

# Load your chatbot model and data
stemmer = LancasterStemmer()

# Provide the full paths for the files
model_path = "C:/Users/haris/Downloads/Deep-Learning-Based-Chatbot-For-Medical-Assistance-master/version 1/chatbot_model.h5"
intents_path = "C:/Users/haris/Downloads/Deep-Learning-Based-Chatbot-For-Medical-Assistance-master/version 1/intents2.json"
words_path = "C:/Users/haris/Downloads/Deep-Learning-Based-Chatbot-For-Medical-Assistance-master/version 1/words.pkl"
labels_path = "C:/Users/haris/Downloads/Deep-Learning-Based-Chatbot-For-Medical-Assistance-master/version 1/labels.pkl"

model = load_model(model_path)
intents = json.loads(open(intents_path).read())
words = pickle.load(open(words_path, 'rb'))
labels = pickle.load(open(labels_path, 'rb'))

# Your chatbot logic
def chatbot_response(text):
    # Implement your chatbot response logic here
    # You can use the existing chatbot_response function in your previous code
    ints = predict_class(text, model)
    response_info = getResponse(ints, intents)
    return response_info

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get user input from the form
    user_input = request.form['user_input']

    # Call your chatbot logic with user_input
    bot_response_info = chatbot_response(user_input)

    # Display the chatbot's response in the HTML template
    return render_template('index.html', user_input=user_input, bot_response=bot_response_info)

if __name__ == '__main__':
    app.run(debug=True)

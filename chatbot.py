import mysql.connector
import nltk
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import random

# Load dataset
with open('responses.json') as file:
    data = json.load(file)

# Preprocessing
nltk.download('punkt')
tokenizer = nltk.word_tokenize

# Prepare training data
patterns = []
labels = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        labels.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

# Tokenize and encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Tokenizing input
all_words = []
x_train = []
y_train = []

for pattern in patterns:
    tokens = tokenizer(pattern)
    all_words.extend(tokens)
    x_train.append(tokens)

y_train = encoded_labels

# Convert words to vectors
unique_words = sorted(set(all_words))
word_index = {w: i for i, w in enumerate(unique_words)}

x_train_encoded = []
for sentence in x_train:
    encoded_sentence = [word_index[w] for w in sentence if w in word_index]
    x_train_encoded.append(encoded_sentence)

x_train_padded = keras.preprocessing.sequence.pad_sequences(x_train_encoded, padding='post')
y_train = np.array(y_train)

# Build model
model = keras.Sequential([
    keras.layers.Embedding(len(unique_words), 16, input_length=len(x_train_padded[0])),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_padded, y_train, epochs=100)

# Connect to MySQL database
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="chatbot_db"
    )

# Get response from DB
def get_response(tag):
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT response FROM responses WHERE tag = %s", (tag,))
    result = cursor.fetchone()
    db.close()
    return result[0] if result else "I'm sorry, I don't understand."

# Chat function
def chat():
    print("Chatbot is running... Type 'quit' to exit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        tokens = tokenizer(user_input)
        encoded_input = [word_index.get(w, 0) for w in tokens]
        padded_input = keras.preprocessing.sequence.pad_sequences([encoded_input], maxlen=len(x_train_padded[0]))
        prediction = model.predict(padded_input)
        tag = encoder.inverse_transform([np.argmax(prediction)])[0]
        
        response = get_response(tag)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()

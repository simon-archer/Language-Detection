import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model('../Training/language_detection')

# Load the tokenizer
with open('../Training/path_to_save_your_tokenizer', 'rb') as f:
    tokenizer = pickle.load(f)

while True:
    # Get a new sentence from the user
    new_sentence = input('Enter a new sentence to be rated (or "q" to quit): ')

    # Quit if the user entered 'q'
    if new_sentence.lower() == 'q':
        break

    # Preprocess the sentence
    new_sentence_seq = tokenizer.texts_to_sequences([new_sentence])
    new_sentence_pad = pad_sequences(new_sentence_seq, maxlen=100)

    # Make a prediction with the model
    prediction = model.predict(new_sentence_pad)

    # Print the rating
    print(f'The rating for the sentence is: {prediction[0][0]}')

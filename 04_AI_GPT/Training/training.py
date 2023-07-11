import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
data = pd.read_csv('../Datasets/cleaned_data.csv')

# Convert all entries in the 'comment_text' column to string
data['comment_text'] = data['comment_text'].astype(str)

# Convert the 'bad' column to integer type
data['bad'] = data['bad'].astype(int)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['comment_text'], data['bad'], test_size=0.2, random_state=0)

# Define the number of words to consider as features
num_words = 30000

# Tokenize the text
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Save the tokenizer in a more friendly format
tokenizer_dict = {
    "word_index": tokenizer.word_index,
    "index_word": tokenizer.index_word,
    "word_counts": tokenizer.word_counts,
}

with open('path_to_save_your_tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer_dict, f, ensure_ascii=False)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# Build the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=num_words, output_dim=32),
    Dropout(0.5),
    tf.keras.layers.LSTM(32),
    Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Train the model
model.fit(X_train_pad, y_train, epochs=10,
          validation_data=(X_test_pad, y_test), callbacks=[early_stopping])

# Save the model
model.save('../Detection/language_detection')

# Convert using this shell command run in the "Detection" folder
# tensorflowjs_converter --input_format=tf_saved_model ./language_detection ./language_detection_tfjs_model

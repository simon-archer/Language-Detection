import pandas as pd
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


class InteractivePlot(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 4))
        plt.ion()

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        self.ax[0].clear()
        self.ax[1].clear()

        self.ax[0].plot(self.losses, label='Training Loss')
        self.ax[0].plot(self.val_losses, label='Validation Loss')
        self.ax[0].legend()
        self.ax[0].set_xlabel('Epoch')
        self.ax[0].set_ylabel('Loss')
        self.ax[0].set_title('Loss Evolution')

        self.ax[1].plot(self.acc, label='Training Accuracy')
        self.ax[1].plot(self.val_acc, label='Validation Accuracy')
        self.ax[1].legend()
        self.ax[1].set_xlabel('Epoch')
        self.ax[1].set_ylabel('Accuracy')
        self.ax[1].set_title('Accuracy Evolution')

        plt.draw()
        plt.pause(0.001)

    def on_train_end(self, logs={}):
        plt.ioff()


# Load the data
data = pd.read_csv('04_AI_GPT/Datasets/cleaned_data_new.csv')


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text,
                  flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove @ and #
    text = re.sub(r'[\W]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r"^\s+", '', text)  # Remove space from the start
    text = re.sub(r"\s+$", '', text)  # Remove space from the end

    # Remove stopwords (optional, can be adjusted later depending on model performance)
    text = ' '.join(word for word in text.split()
                    if word not in ENGLISH_STOP_WORDS)

    return text


def handle_negation(text):
    negation_words = ['not', 'ikke', 'nei', 'no',
                      'aldri', 'never', 'uten', 'without']
    negation_suffix = '_NEG'
    in_negation = False
    negated_text = []

    for word in text.split():
        if word in negation_words:
            in_negation = not in_negation
            negated_text.append(word)
        elif re.match(r'[.!?]', word):
            in_negation = False
            negated_text.append(word)
        elif in_negation:
            negated_text.append(word + negation_suffix)
        else:
            negated_text.append(word)

    return ' '.join(negated_text)


# Convert all entries in the 'comment_text' column to string
data['comment_text'] = data['comment_text'].astype(str)

# Convert the 'bad' column to integer type
data['bad'] = data['bad'].astype(int)

# Clean the text and handle negation
data['cleaned_text'] = data['comment_text'].apply(clean_text)
data['negated_text'] = data['cleaned_text'].apply(handle_negation)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['negated_text'], data['bad'], test_size=0.2, random_state=0)

# Tokenization and Padding
max_features = 30000  # or adjust as needed
maxlen = 100  # or adjust as needed

tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Save the tokenizer in a more friendly format
tokenizer_dict = {
    "word_index": tokenizer.word_index,
    "index_word": tokenizer.index_word,
    "word_counts": tokenizer.word_counts,
}

with open('path_to_save_your_tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer_dict, f, ensure_ascii=False)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# Build the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=max_features, output_dim=32),
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

# Define live plotting
live_plot = InteractivePlot()

# Train the model
history = model.fit(X_train_pad, y_train, epochs=10,
                    validation_data=(X_test_pad, y_test),
                    callbacks=[early_stopping, live_plot])

# Save the model
model.save('../Detection/language_detection')

# Convert the TensorFlow model to .JS using this shell command run in the "Detection" folder
# tensorflowjs_converter --input_format=tf_saved_model ./language_detection ./language_detection_tfjs_model

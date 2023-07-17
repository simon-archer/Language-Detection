
# Norwegian Sentiment Detection

Language Detection is a client-side sentiment analysis application specifically designed for Norwegian text. Leveraging TensorFlow.js for in-browser machine learning, this application enables real-time language detection directly in the user's browser.

One of the key features of this tool is its GDPR-compliant mechanism. As the sentiment analysis is performed locally on the user's machine, the data doesn't leave the user's browser, ensuring data privacy and compliance with data protection regulations.

## Project Overview

The application uses an input field to accept user Norwegian sentences, runs sentiment analysis using a trained TensorFlow.js model, and returns sentiment scores. It also includes functionalities to manually label sentence sentiment and export the annotated dataset for further usage.

This project utilizes a dataset from the Jigsaw Toxic Comment Classification Challenge on Kaggle, which has been translated into Norwegian. The dataset is released under the CC0 license, and Wikipedia's CC-SA-3.0 governs the underlying comment text.

GPT-3.5 Turbo has generated additional samples.


> Please note that the dataset has approximately:
> 85.63% "good" comments (represented by 0 in the 'bad' column)
> 14.37% "bad" comments (represented by 1 in the 'bad' column)
> This indicates a somewhat imbalanced dataset, with significantly more "good" comments than "bad" ones.

## Setup and Usage

### 1. Clone the Repository

Clone the repository to your local machine using the Language Detection tool.

`git clone https://github.com/simon-archer/Language-Detection.git` 

### 2. Serve the HTML File

Use an HTTP server to serve the HTML file. For instance, if you have Python installed, you can use its built-in HTTP server:

`python -m http.server` 

For Python 3, use:

`python3 -m http.server` 

Next, open your web browser and navigate to `localhost:8000` or your server's port.

### 3. Analyse Sentence Sentiment

Type a Norwegian sentence into the input field and click the 'Predict' button. The sentiment of the sentence, predicted by the model, will be displayed.

### 4. Label the Sentiment

After a prediction, you can manually label the sentence as 'Good' or 'Bad'. This action adds the sentence and its sentiment label to the dataset and updates the counter.

### 5. Download the Dataset

Once you've marked multiple sentences, click the 'Download' button to save the dataset. This data will be in CSV format, named 'data.csv'.

## Important Note

The paths to the model and tokenizer are currently set as 'Detection/language_detection_tfjs_model/model.json' and 'Training/path_to_save_your_tokenizer.json.' Ensure these files are placed correctly or modify the paths to align with your directory structure.

## Contributing

We welcome contributions! Your input is appreciated, whether it's bug reports, feature requests, or new PRs. Start a thread in the 'Issues' section for questions or discussions.

Remember, this tool's main focus is ensuring GDPR compliance by analyzing the client-side. Any enhancement should keep this primary focus in mind.

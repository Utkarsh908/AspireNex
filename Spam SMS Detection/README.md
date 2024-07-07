# SMS Spam Detection

This project builds an AI model to classify SMS messages as spam or legitimate. We use techniques like TF-IDF for feature extraction and classifiers like Naive Bayes, Logistic Regression, and Support Vector Machines (SVM) to identify spam messages. The dataset is sourced from Kaggle.

## Dataset

The dataset used for this project is the SMS Spam Collection Dataset available on Kaggle. It contains a collection of SMS messages labeled as "ham" (legitimate) or "spam".

## Requirements

To run this project, you'll need the following Python libraries:
- pandas
- scikit-learn
- joblib

You can install these libraries using pip:
```bash
pip install pandas scikit-learn joblib
```
## Instructions
1.Download the dataset:
Download the dataset from Kaggle and save it as spam.csv in the project directory.

2.Run the script:
Execute the script sms_spam_detection.py to preprocess the data, train the models, evaluate their performance, and save the best-performing model.

3.Classify new messages:
Use the saved model to classify new SMS messages as spam or not spam.

## Script Overview
- Data Preparation :
  First, the dataset is loaded and preprocessed. We remove punctuation, convert the text to lowercase, and remove extra spaces. The labels are encoded as binary
  values (0 for ham and 1 for spam). The data is then split into training and test sets.
- Feature Extraction : 
  We use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text data into numerical features.
- Model Building : 
  We train three models on the extracted features:
  Naive Bayes, 
  Logistic Regression, 
  Support Vector Machines (SVM).
- Deployment :The best-performing model is saved using The best-performing model is saved using 'joblib'. A simple application is provided to classify new
  SMS messages using the saved model.. A simple application is provided to classify new SMS messages using the saved model.
## Usage
1. Training the Model:
Run the script sms_spam_detection.py to train the models and save the best-performing model.

2. Classifying New Messages:
Use the following code to classify new SMS messages:
```bash
import joblib

# Load the model and vectorizer
loaded_model = joblib.load('sms_spam_classifier.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Classify a new message
new_message = ["Congratulations! You've won a free ticket to the Goa. Call now!"]
new_message_tfidf = loaded_vectorizer.transform(new_message)
prediction = loaded_model.predict(new_message_tfidf)
print("Spam" if prediction[0] else "Not Spam")
```
## Results
After training and evaluating the models, the Logistic Regression model performed the best with the following metrics:

- Accuracy: 0.968609865470852
- Precision: 0.9914529914529915
- Recall: 0.7733333333333333
- F1-Score: 0.8689138576779026

## Acknowledgments
- The dataset is provided by UCI Machine Learning Repository and can be found on Kaggle.
```bash
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
```

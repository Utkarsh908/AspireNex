# Credit Card Fraud Detection

This repository contains code for detecting fraudulent credit card transactions using machine learning models. The dataset used for this project is from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Project Description

The goal of this project is to build a model to classify credit card transactions as fraudulent or legitimate. We experimented with various algorithms such as Logistic Regression, Decision Trees, and Random Forests.

## Dataset

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **V1, V2, ..., V28**: Principal components obtained with PCA.
- **Amount**: Transaction amount.
- **Class**: 1 for fraudulent transactions, 0 otherwise.

## Setup

### Prerequisites

- Python 3.x
- Required Python libraries (listed in `requirements.txt`)


### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Utkarsh908/AspireNex/tree/aa017cf28c4f975266b06b499c3c54e33c2d4be3/Credit%20Card%20Fraud%20Detection
   cd credit-card-fraud-detection
2. Create a virtual environment and activate it:
  ```sh
 python -m venv venv
 source venv/bin/activate   # On Windows, use `venv\Scripts\activate
 ```
3. Install the required packages:
 ```
pip install -r requirements.txt
```
4. Download the dataset from Kaggle and place it in the repository directory.

## Models

In this project, we experimented with several machine learning algorithms to detect fraudulent credit card transactions. Below is a detailed description of each model used, along with their performance metrics.

### 1. Logistic Regression

**Description:**  
Logistic Regression is a statistical model used for binary classification problems. It estimates the probability of a binary outcome based on one or more predictor variables.

**Performance Metrics:**
| Metric        | Score   |
|---------------|---------|
| Accuracy      | 99.92%  |
| Precision     | 84.38%  |
| Recall        | 62.79%  |
| F1-Score      | 72.08%  |


### Decision Tree Classifier

**Description:** 
The Decision Tree Classifier uses a tree-like model of decisions and their possible consequences. It splits the data into subsets based on feature values to predict the target variable.

**Performance Metrics:**
| Metric        | Score   |
|---------------|---------|
| Accuracy      | 99.93%  |
| Precision	    | 88.57%  |
| Recall	      | 67.44%  |
| F1-Score	    | 76.62%  |

## Conclusion
Among the models tested, Random Forest Classifier performed the best with the highest accuracy, precision, recall, and F1-score. It is the recommended model for detecting fraudulent transactions in this dataset.



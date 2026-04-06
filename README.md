# SpamDetector
# Spam Detector 🛡️

A machine learning model that classifies SMS messages as spam or 
legitimate using Naive Bayes algorithm, achieving 99.19% accuracy.

## Features
- Trained on 5500+ real SMS messages
- 99.19% classification accuracy
- Real-time prediction on custom messages
- Error analysis on edge cases

## Technologies Used
- Python
- scikit-learn
- pandas
- CountVectorizer
- Multinomial Naive Bayes

## How to Run
1. Clone this repository
2. Install dependencies:
   pip install pandas scikit-learn
3. Download the SMS Spam dataset from:
   https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
4. Rename the file to spam.csv and place in project folder
5. Run:
   python spam_detector.py

## Sample Output
Message: Congratulations! You won a free iPhone. Click here!
Prediction: SPAM
Message: Hey are you coming to college tomorrow?
Prediction: HAM
## Model Performance
- Algorithm: Multinomial Naive Bayes
- Training data: 80% (4457 messages)
- Testing data: 20% (1115 messages)
- Accuracy: 99.19%

## What I Learned
- Machine learning pipeline from data to prediction
- Text vectorization using CountVectorizer
- How Naive Bayes works for text classification
- Error analysis and model limitations

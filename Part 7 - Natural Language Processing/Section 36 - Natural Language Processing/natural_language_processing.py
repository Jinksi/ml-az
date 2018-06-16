# Natural Language Processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Use tsv rather than csv to allow commas in the text
# Ignore quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
dataset

# Cleaning the texts
# - remove non-relevant words (the, on, and) (stopwords)
# - remove punctuation
# - remove numbers unless significant
# - stemming: simplify versions of same word (love, loved)
# - lowercase
# - tokenisation: split into different words and attribute number of times word occur in a review

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_text(str):
    # keep only letters with regex
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=str)
    # split into single lowercase words
    text = text.lower().split()
    # create stemmer
    ps = PorterStemmer()
    # fancy inline for loop with if not statement
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    # join back into string
    text = ' '.join(text)
    return text

corpus = []
for i in range(0, len(dataset)):
    text = dataset['Review'][i]
    text = clean_text(text)
    corpus.append(text)

# Create the Bag of Words model
# a matrix of words (columns) and reviews (rows)
# will have a lot of 0 values, sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(raw_documents=corpus).toarray()
y = dataset['Liked'].values

# Naive Bayes classifier
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
len(X_test)
len(y_test)

# Fitting Naive Bayes to the Training
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X=X_train, y=y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred

# Making the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
report = classification_report(y_test, y_pred)
print(report)
# Precision (exactness)
# Recall (completeness)
# F1 Score (compromise precision/recall)
score = accuracy_score(y_test, y_pred)
print(score)

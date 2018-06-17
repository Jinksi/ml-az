# Artificial Neural Network

# - dataset is a list of a bank's customers
# - customers are leaving at an increasing churn rate
# - various customer features are provided
# - dependent variable == exited within the 6month time period

# Just disables the tensorflow AVX CPU warning
# https://stackoverflow.com/questions/47068709/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
#
# Part 1 - Data preprocessing ( from classification template)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset
# Select relevant features: ignore columns 0,1,2
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, [13]].values
X
# Encoding Dummy Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Country column
encoder_X_1 = LabelEncoder()
X[:, 1] = encoder_X_1.fit_transform(X[:, 1])
# Gender column
encoder_X_2 = LabelEncoder()
X[:, 2] = encoder_X_1.fit_transform(X[:, 2])
X
# OneHotEncoding
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
# Dummy variable trap - remove first dummy variable
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
len(X_test), len(X_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#
#
# Part 2 - make the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialise the ANN
classifier = Sequential()

# adding layers
# hidden layer nodes/units rule of thumb: average of input nodes and output nodes
# relu = rectifier function
hidden_layer = Dense(units=6, kernel_initializer='uniform', activation='relu')
classifier.add(hidden_layer)
hidden_layer_2 = Dense(units=6, kernel_initializer='uniform', activation='relu')
classifier.add(hidden_layer_2)
# output activation function will use sigmoid function for probability
output_layer = Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
classifier.add(output_layer)

# Compile the ANN with "Adam SGD"
# Use logarithmic loss function
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
history = classifier.fit(x=X_train, y=y_train, batch_size=10, epochs=100)

#
#
# Part 3 - make predictions and evaluate

acc_over_epochs = history.history['acc']
plt.scatter(range(0, len(acc_over_epochs)), acc_over_epochs)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# convert to boolean
y_pred = y_pred > 0.5

# Making the confusion matrix
# https://en.wikipedia.org/wiki/Confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
cm
print(classification_report(y_test, y_pred))

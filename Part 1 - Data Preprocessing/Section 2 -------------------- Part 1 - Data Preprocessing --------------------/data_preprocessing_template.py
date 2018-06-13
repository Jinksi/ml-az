# Data Preprocessing

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv')

# Create matrix of features (independent variables)
# iloc[all lines, all columns except last]
X = dataset.iloc[:, :-1].values

# Create dependent variable vector
# include only last column
y = dataset.iloc[:, 3]


# Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# feed all lines, only columns [1:2] that have missing values
imputer = imputer.fit(X[:, 1:3])
# replace X column values with imputer.fit values
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# only first ( country ) column
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Dummy variable encoding
# set categorical_feature indice / column to the Country column
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# y has only 1 column 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the data into Training / Test set
from sklearn.model_selection import train_test_split
# random_state = random seed value for reproducible split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Fit to train data
X_train = sc_X.fit_transform(X_train)
# Transform is based on train data fitted scale
X_test = sc_X.transform(X_test)
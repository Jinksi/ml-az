# Data Preprocessing

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv')

# Create matrix of features (independent variables)
# [all lines, all columns except last]
X = dataset.iloc[:, :-1].values
# Create dependent variable vector
y = dataset.iloc[:, 3]


# Splitting the data into Training / Test set
from sklearn.model_selection import train_test_split
# random_state = random seed value for reproducible split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Fit to train data
X_train = sc_X.fit_transform(X_train)
# Transform is based on train data fitted scale
X_test = sc_X.transform(X_test)"""
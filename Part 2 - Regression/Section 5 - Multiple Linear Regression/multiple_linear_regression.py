# Multiple Linear Regression

# Profit | R&D Spend | Admin | Marketing | State (Categorical)
# State needs Dummy Variables
# Only include first Dummy Variable which explains both categories

# Dummy variable trap aka multicollinearity

# y = b0(x0) + b1*x1 + b2*x2 + b3+x3  + bn*Xn

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Dummy variable encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# remove first column (first dummy variable column)
X = X[:, 1:]
# usually handled by the library

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# handled by sklearn linear_model library

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# Add x0 = 1 column to maintain b0 constant
# X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X = np.append(arr = np.ones((50, 1)), values = X, axis = 1)
X
# Significance Level = 0.05
# Fit will all possible predictors

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# ols = Ordinary Least Squares
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
# find P Value in summary
regressor_ols.summary()

# Remove index with highest P value
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

# Remove index with highest P value
X_opt = X[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

# Remove index with highest P value
# Column 3 (R & D spend) has a significant statistical impact of the independent variable (profit)
X_opt = X[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

# Remove index with highest P value
X_opt = X[:, [0, 3]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_ols.summary())

# Split and train on X_opt
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# handled by sklearn linear_model library

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_opt_pred = regressor.predict(X_test)
y_opt_pred

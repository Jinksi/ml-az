# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# X should be a matrix,[[1], [2], [3], ...]
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# No data split, perfectly matching the dataset

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Create multi linear regression with new matrix of variables
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualise lin reg
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X))
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
# Predict 6.5 level salary
y_pred = lin_reg.predict(6.5)
plt.scatter(6.5, y_pred)
plt.show()

# Visualise poly reg
# New X_grid for plotting regression curve
X_grid =np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)))
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
# Predict 6.5 level salary
y_pred_2 = lin_reg_2.predict(poly_reg.fit_transform(6.5))
plt.scatter(6.5, y_pred_2)
plt.show()

# Predicting new result
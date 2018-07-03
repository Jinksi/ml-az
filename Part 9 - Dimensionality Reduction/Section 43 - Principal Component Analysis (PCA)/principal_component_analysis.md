# Principal Component Extraction

- Identify correlations between variables
- Reduce dimensions of a d-dimensional dataset by projecting it onto a k-dimensional subspace (where k < d)
- Unlike linear regression which predicts new values, PCA attempts to quantify the relationships between X and Y values, finding a list of principal axes
- Considered an _unsupervised model_ due to the fact that the dependent variable is not considered

#### Steps

- Standardise the data
- Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition
- Sort eigenvalues in descending order, selecting k eigenvectors that correspond to the k largest eigenvalues, where k is the number of dimensions in the new feature subspace (k < D)
- Construct the projection matrix W from the selected k eigenvectors
- Transform the original dataset X via W to obtain a k-dimensional feature subspace Y

#### More

- [plot.ly](https://plot.ly/ipython-notebooks/principal-component-analysis/)
- [Setosa](http://setosa.io/ev/principal-component-analysis/)

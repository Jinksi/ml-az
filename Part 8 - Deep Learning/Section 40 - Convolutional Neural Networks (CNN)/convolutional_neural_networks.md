# Convolutional Neural Network

- the human brain will process images by identifying features
- Yann Lecun - godfather of CNN
- images are an array of pixels
  - B/W Image 2x2px
    - 2d array brightness 0-255 x each pixel
  - Colour Image 2x2px
    - 3d array RGB x 0-255 x each pixel

## Step 1: Convolution
- a _Feature Detector / Filter_ is a subset (matrix) of pixel values (3x3)
- multiply the input image pixel data matrix through the feature detector matrix
- the result is placed in a new matrix with the sum of multiplied values
- each stride (new starting x,y pixel) taken is a new value in the matrix
- the result is a _Feature Map_, preserving only the features of the image
- multiple Feature Maps will be created to identify different features

#### ReLU Layer
- (vague explanation)
- apply the rectifier function (rectified linear unit) to the feature map values
- results in only positive values, breaking up the linearity
- [More Info](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

## Step 2: Max Pooling Layer
- _Spatial Variance_
  - flexibility to find features with different image conditions: transformations, rotations, texture, etc
- Pooling == downsampling
- for each _Feature Map_ create a _Pooled Feature Map_ keeping max value of a pixel matrix (2x2px)
- simplify the Feature Map further
- reduces overfitting
- Other methods: mean pooling, sum pooling, subsampling...
- [Evaluation of Pooling Operations in
Convolutional Architectures for Object
Recognition](http://www.ais.uni-bonn.de/papers/icann2010_maxpool.pdf)
- [Interactive example](http://scs.ryerson.ca/~aharley/vis/conv/flat.html)

## Step 3: Flattening

- Take the _Pooled Feature Map_ and flatten it into a vector of inputs for an ANN

## Step 4: Full Connection

- The hidden layers of the CNN are fully connected (dense)
- Weight are calculated and improved via back-propagation

## Additional Reading
  - [The 9 Deep Learning Papers You Need To Know About ](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)

# Convolutional Neural Network

# Just disables the tensorflow AVX CPU warning
# https://stackoverflow.com/questions/47068709/
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
#
# Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Initialise the CNN
classifier = Sequential()

# Step 1 - Convolution
# Input Image -> Feature Detectors (filters) -> Feature Maps
convolution_layer = Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu')
classifier.add(convolution_layer)

# Step 2 - Max Pooling
pooling_layer = MaxPooling2D(pool_size=(2, 2))
classifier.add(pooling_layer)

# Add a second convolution_layer
# remove input_shape, inferred from previous layer
convolution_layer_2 = Convolution2D(filters=32, kernel_size=(3, 3), activation='relu')
classifier.add(convolution_layer_2)
pooling_layer_2 = MaxPooling2D(pool_size=(2, 2))
classifier.add(pooling_layer_2)

# Step 3 - Flattening
flattening_layer = Flatten()
classifier.add(flattening_layer)

# Step 4 - Full Connection
# experiment to find nodes in layer
hidden_layer = Dense(units=256, activation='relu')
classifier.add(hidden_layer)
# sigmoid for probability of output
output_layer = Dense(units=1, activation='sigmoid')
classifier.add(output_layer)

# Compile the CNN
# Using the asam optimizer function
# Use categorical_crossentropy for more classification outputs
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#
#
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
# Use data augmentation during image preprocessing to enrich dataset with more image training data
train_datagen = ImageDataGenerator(
    rescale=1./255, # map brightness values 0-1
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(128, 128), # resize to the CNN input shape
    batch_size=32,
    class_mode='binary') # if more than 2 classes update class_mode

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    #steps_per_epoch=8000, # images in training set
    epochs=25,
    validation_data=test_set)
    #validation_steps=2000) # images in test set

from datetime import datetime
i = datetime.now()
filename = f'cnn_model_{i.strftime("%Y-%m-%d-%H-%M-%S")}.h5'
classifier.save(filename)
print(f'Saved file: {filename}')

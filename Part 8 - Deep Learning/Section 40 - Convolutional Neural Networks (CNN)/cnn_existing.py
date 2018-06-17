# Convolutional Neural Network

# load existing model
from keras.models import load_model
classifier = load_model('cnn_model_2018-06-17-18-15-45.h5')

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
    target_size=(64, 64), # resize to the CNN input shape
    batch_size=32,
    class_mode='binary') # if more than 2 classes update class_mode

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    #steps_per_epoch=8000, # images in training set
    epochs=2,
    validation_data=test_set)
    #validation_steps=2000) # images in test set

from datetime import datetime
i = datetime.now()
filename = f'cnn_model_{i.strftime("%Y-%m-%d-%H-%M-%S")}.h5'
classifier.save(filename)
print(f'Saved file: {filename}')

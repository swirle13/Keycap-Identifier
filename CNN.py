# Convolutional Neural network

# Importing the Keras libraries and packages
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl
from IPython.display import display
from PIL import Image
import numpy as np

# In order to fight overfitting, I am increasing the entropic capacity of my
# model. The main way to modulate entropic capacity is the choice of the number
# of parameters in my model, i.e. the number of layers and the size of each
# layer. I'm using a convnet with few layers and few filters per layer, along
# with data augmentation and dropout. Dropout also helps overfitting by
# preventing a layer from seeing the exact same pattern twice, thus acting
# in a way analaguous to data augmentation.
#
# This is a simple stack of 3 convolution layers with a ReLU activation and
# followed by max-pooling layers.
#
# On top, I stick two fully connected layers and end the model with a single
# unit and a softmax activation for my categorical classification.

#Initalize the CNN
classifier = Sequential()

# Convolution and Pooling of multiple layers
classifier.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# model now outputs 3D feature maps (height, width, features)

# Flattening and connection
classifier.add(Flatten()) # converts 3D feature maps to 1D feature vectors
classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(5))
classifier.add(Activation('softmax')) # previously sigmoid

# compiling the CNN
# use categorical_crossentropy for multiple categories
# targets should be in categorical format, i.e. 10 cats means 10D vector
# with all zeros except for a 1 at the index corresponding to class of the sample
# In order to convert integer targets to categorical targets, use Keras utility
# 'to_categorical'
# from keras.utils import to_categorical
# categorical_labels = to_categorical(int_labels, num_classes = None)

# was using optimizer 'rmsprop' but am currently testing SGD to prevent
# a bouncing back and forth of accuracy. Tried Nadam, SGD, and adam and adam has
# turned out to be the most accurate with the options and layers that I'm using
classifier.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

batch_size = 32

# Fitting the CNN to the images
train_datagen = ImageDataGenerator(
    rotation_range = 90,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    'keycapdata/training_set',
    target_size = (150, 150),
    batch_size = batch_size,
    # class_mode = 'binary' # categorical with more than 2 types
    # default mode is 'categorical' so if commented out, using more than 2 labels
)

validation_set = test_datagen.flow_from_directory(
    'keycapdata/validation_set',
    target_size = (150, 150),
    batch_size = batch_size,
    # class_mode = 'binary'
)

classifier.fit_generator(
    training_set,
    steps_per_epoch = 20,
    epochs = 10,
    validation_data = validation_set,
    validation_steps = 10
)

# save model as .h5 aka .hdf5, Hierarchical Data Format.
classifier.save('keycapidentifier.h5')

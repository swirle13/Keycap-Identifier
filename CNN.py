# Convolutional Neural network

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initalize the CNN
classifier = Sequential()

# step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))

# step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# step 3 - Flattening
classifier.add(Flatten())

# step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    'keycapdata/training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

test_set = test_datagen.flow_from_directory(
    'keycapdata/test_set',
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'binary'
)

from IPython.display import display
from PIL import Image

classifier.fit_generator(
    training_set,
    steps_per_epoch = 3, # was 8000, shortening for laptop
    epochs = 2,
    validation_data = test_set,
    validation_steps = 1 # was 800
)

import numpy as  np
from keras.preprocessing import image
test_image = image.load_img('keycapdata/test_set/pulse/t01.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
# print(training_set.class_indicies)

print("result:", result)
if result[0][0] >= 0.5:
    prediction = '\nMy best guess is GMK Carbon\n'
else:
    prediction = '\nMy best guess is SA Pulse\n'

print(prediction)

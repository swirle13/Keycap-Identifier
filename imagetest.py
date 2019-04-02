# Python script to test files against the trained tensorflow model

import tensorflow as tf
import numpy as np
# from keras import backend as K
from keras.models import Sequential, load_model
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
# from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.python.saved_model import builder as saved_model_builder
# from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl
# from IPython.display import display
from PIL import Image
# from glob import glob

batch_size = 16

classifier = load_model('keycapidentifier.h5')

# print(classifier.outputs)
# print(classifier.inputs)

test_image = image.load_img('keycapdata/test_set/1976/t02.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict_classes(test_image)

# class_names = glob("keycapdata/training_set/*") # reads all the folders with images
# class_names = sorted(class_names)
# name_id_map = dict(zip(class_names, range(len(class_names))))
# print(name_id_map)

# this is code to print out the labels of the training data
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

generator = train_datagen.flow_from_directory(
    'keycapdata/test_set', batch_size=batch_size)
label_map = generator.class_indices
print(label_map)

print("result:", result)

for key, value in label_map.items():
    if result == value:
        print("My best guess is GMK/SA", key)

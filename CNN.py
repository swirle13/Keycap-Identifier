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

# sess = tf.Session()
# K.set_session(sess)
# K.set_learning_phase(0)

model_version = "2"

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

# Flattening and connection
classifier.add(Flatten())
classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))

# compiling the CNN
# use categorical_crossentropy for multiple categories
# targets should be in categorical format, i.e. 10 cats means 10D vector
# with all zeros except for a 1 at the index corresponding to class of the sample
# In order to convert integer targets to categorical targets, use Keras utility
# 'to_categorical'
# from keras.utils import to_categorical
# categorical_labels = to_categorical(int_labels, num_classes = None)
classifier.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

batch_size = 16

# Fitting the CNN to the images
train_datagen = ImageDataGenerator(
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
    class_mode = 'binary' # categorical with more than 2 types
)

validation_set = test_datagen.flow_from_directory(
    'keycapdata/test_set',
    target_size = (150, 150),
    batch_size = batch_size,
    class_mode = 'binary'
)

classifier.fit_generator(
    training_set,
    steps_per_epoch = 10, # was 8000, shortening for laptop
    epochs = 10,
    validation_data = validation_set,
    validation_steps = 10 # was 800
)

classifier.save('first_try.h5')

# prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
# {"inputs": training_set}, {"prediction":test_set})
#
# valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(prediction_signature)
# if(valid_prediction_signature == False):
#     raise ValueError("Error: Prediction signature not valid!")
#
# builder = saved_model_builder.SavedModelBuilder('./'+model_version)
# legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
# builder.add_meta_graph_and_variables(
#     sess, [tag_constants.SERVING],
#     signature_def_map={
#         signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature,
#     },
#     legacy_init_op=legacy_init_op)
#
# builder.save()

#
# test_image = image.load_img('keycapdata/test_set/carbon/t01.jpg', target_size = (150, 150))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# # print(training_set.class_indicies)
#
# print("result:", result)
# if result[0][0] >= 0.5:
#     prediction = '\nMy best guess is SA Pulse\n'
# else:
#     prediction = '\nMy best guess is GMK Carbon\n'

# print(prediction)

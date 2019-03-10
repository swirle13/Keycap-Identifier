# test image

from keras.models import load_model
# from PIL import Image

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

import webbrowser

classifier = load_model('first_try.h5')

test_image = image.load_img('keycapdata/test_set/carbon/t05.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
# print(training_set.class_indicies)

# print("result:", result)

search_terms = []
if result[0][0] >= 0.5:
    prediction = '\nMy best guess is SA Pulse\n'
    setting = 1
else:
    prediction = '\nMy best guess is GMK Carbon\n'
    setting = 2

print(prediction)
# chrome_browser = webbrowser.get('/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe')
#
# if setting == 1:
#     url = "https://www.google.com/search?q=SA+Pulse&tbm=isch"
#     chrome_browser.open_new_tab(url)
# if setting == 2:
#     url = "https://www.google.com/search?q=GMK+Carbon&tbm=isch"
#     chrome_browser.open_new_tab(url)

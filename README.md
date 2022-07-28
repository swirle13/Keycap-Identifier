# Keycap-Identifier
Image Processing to Recognize Keycap Sets

This is a personal project utilizing Tensorflow, Keras, Anaconda, and the suite of other supporting applications in order to identify keyboard keycap sets.

This is a Flutter application that uses a trained TF model to pass images to the model to analyze against a collection of keycap data and recognize what set the given image is.

Currently, the TF doesn't interact with the app and are two separate pieces, but as these continue to be worked on, it will become one cohesive project.

Please set up a python venv with python 3.7.1 or greater, as this project has dependencies that require python 3.7.x.

To set up the venv for this, us the command `virtualenv --no-site-packages --distribute venv && source venv/bin/activate && pip install -r requirements.txt`

I highly recommend using this [github repository](https://github.com/lakshayg/tensorflow-build) for installing a version of tensorflow compiled with your specific CPU to utilize the AVX, AVX2, etc. This will utilize the large-instruction-size caches and increase efficiency and speed of convolution, dot-product, and more.

# Screenshots
Below are some screenshots of the app. It is still in an MVP state, functionality-wise, as I learn more about Flutter. The trained model for recognizing the images works on its own which you can use via CLI if you'd like to test it.

Home page | Camera | Submit Image
:--------:|:------:|:-----------:
![messages_0 (2)](https://user-images.githubusercontent.com/9778474/181602212-a8d072ce-bd16-47a2-b8dc-6a5318de22aa.jpeg) | ![messages_0 (3)](https://user-images.githubusercontent.com/9778474/181602314-b67dad4a-a94b-4811-b6c2-98f8b9934617.jpeg) | ![messages_0 (4)](https://user-images.githubusercontent.com/9778474/181602321-58c975d1-02ea-4df4-8c31-7e3e87c9c01f.jpeg) 

# Case Study of Keras 
## Technology and Platform
Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow, Microsoft 
Cognitive Toolkit, Theano, or PlaidML. Designed to enable fast experimentation with deep neural networks, it focuses on 
being user-friendly, modular, and extensible. 

As we know, the fundaments of TensorFlow framework was written is C++ while Keras was written in Python. As a result, Ke
ras was conceived to be an interface rather than a standalone machine-learning framework. It offers a higher-level, more
intuitive set of abstractions that make it easy to develop deep learning models regardless of the computational backend
used. Below are the enabled version of Python and backend for Keras.

- python: 3.6
env: KERAS_BACKEND=tensorflow 
TEST_MODE=INTEGRATION_TESTS PIL=Pillow
- python: 3.6
env: KERAS_BACKEND=tensorflow 
TEST_MODE=PEP8_DOC PIL=Pillow
- python: 3.6
env: KERAS_BACKEND=tensorflow TEST_MODE=API
- python: 2.7
env: KERAS_BACKEND=tensorflow
- python: 3.6
env: KERAS_BACKEND=tensorflow
- python: 2.7
env: KERAS_BACKEND=theano THEANO_FLAGS=optimizer=fast_compile MKL="mkl mkl-service" RUN_ONLY_BACKEND_TESTS=1
- python: 3.6
env: KERAS_BACKEND=theano THEANO_FLAGS=optimizer=fast_compile MKL="mkl mkl-service"
- python: 2.7
env: KERAS_BACKEND=cntk PYTHONWARNINGS=ignore RUN_ONLY_BACKEND_TESTS=1
- python: 3.6
env: KERAS_BACKEND=cntk PYTHONWARNINGS=ignore
For implementation, we import Keras library from TensorFlow backend in a Python script . We can build the program in IDE
like PyCharm or Spyder. For data visualization and machine learning, we can use the open-source web-based computing 
environment Jupyter Notebook, which I believe is the best platform so far. Of course, we can simply run python xxx.py 
in the command line shell. But it is not a good way to train a deep neural network using you own CPU since the limit of
 performance. So may want to utilize some cloud computing server like SCC.

## Testing
In the keras/tests repository, they write test scripts to test different modules in Keras. These modules include backend,
datasets, engine, layers, utils, wrappers. They also test important functions like activations, initializer, optimizer, 
metrics, etc.

[test of keras](https://github.com/keras-team/keras/tree/master/tests/keras)

The engineers are proving their testing quality all the time. For example ,in the test_api.py, they import pyux to deal 
with the exception while changing and overwriting the API.

[test.api](https://github.com/keras-team/keras/blob/master/tests/test_api.py)

Keras use Travis-CI for Continuous Integration. Travis CI can test on multiple operating systems, Linux and macOS. I also 
found in their github that skip the CNTK install in some travis jobs to speed up tests. 

[.travis](https://github.com/keras-team/keras/tree/master/.travis)

## Software Architecture
All the functionality and modules are defined in the keras/keras repository. If you want to add or edit functionality, 
just edit the corresponding module. For example, in the layers/convolutional.py, you can edit the convolutional neural 
networks in the entire process. As the core function of Keras or TenserFlow framework, if we can make improvement to that
 part. It would be a great leap for the entire industry of Machine Learning. 
 
As Keras mainly serves as a python library, we can simply import it in python script like: ‘import keras’ or 
‘from Tensorflow import keras’ .

Keras is thread safe so we can use Python and Keras to build a neural network that will load and process multiple sources
of information in parallel. In fact, in reinforcement learning there is an algorithm called Asynchronous Advantage Actor
Critics (A3C) where each agent relies on the same neural network to tell them what they should do in a given state.
In other words, each thread calls model.predict concurrently.

Here is a diagram of using Keras functional API to build a simple deep learning network.

[Linear Diagram]()

First of all, we must create and define a standalone input layer that specifies the shape of input data. Then we create 
a hidden layer as a Dense that can receive input only from the input layer. In this way of connecting layers piece by 
piece that gives the functional API great flexibility. After that, we use the Model class to create a model from the 
existing layers. Below is the code sample.

from keras.models import Model<br>
from keras.layers import Input<br>
from keras.layers import Dense<br>
visible = Input(shape=(10,))<br>
hidden1 = Dense(10, activation='relu')(visible)<br>
hidden2 = Dense(20, activation='relu')(hidden1)<br>
hidden3 = Dense(10, activation='relu')(hidden2)<br>
output = Dense(1, activation='sigmoid')(hidden3)<br>
model = Model(inputs=visible, outputs=output)<br>

Assume we want to define a convolutional neural network for image classification. The model receives 64*64 images as 
inputs, then has a sequence of two convolutional layers and two pooling layers as feature extractors, followed by a 
fully connected layer to interpret the features and an output layer with a sigmoid activation for two-class predictions.
Below is the Python code and model diagram.

from keras.models import Model<br>
from keras.layers import Input<br>
from keras.layers import Dense<br>
from keras.layers import Flatten<br>
from keras.layers.convolutional import Conv2D<br>
from keras.layers.pooling import MaxPooling2D<br>
visible = Input(shape=(64,64,1))<br>
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)<br>
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)<br>
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)<br>
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)<<br>
flat = Flatten()(pool2)<br>
hidden1 = Dense(10, activation='relu')(flat)<br>
output = Dense(1, activation='sigmoid')(hidden1)<br>
model = Model(inputs=visible, outputs=output)<br>


[CNN Diagram]()

## Issue
when I try to load weights of shallower layers as initialization for deeper one, it fails.

    input_tensor = Input(shape=(SIZE[0], SIZE[1], 3))
    out = Conv2D(32, kernel_size=3, strides=1, activation='elu', padding='same', name='encoder_1')(input_tensor)
    out = Conv2D(64, kernel_size=5, strides=2, activation='elu', padding='same', name='encoder_2')(out)
    out = Conv2D(128, kernel_size=5, strides=2, activation='elu', padding='same', name='encoder_3')(out)
    out = Conv2D(256, kernel_size=5, strides=2, activation='elu', padding='same', name='encoder_4')(out)
    out = Conv2D(256, kernel_size=5, strides=2, activation='elu', padding='same', name='encoder_5')(out)
    out = Conv2D(24, kernel_size=5, strides=1, activation='elu', padding='same', name='encoder_6')(out)
    out = Flatten()(out)
    out = Dense(z_size, activation='linear', name='bottleneck', activity_regularizer=l1(regul_const))(out)
    out = Dense(9 * 16 * 24, activation='elu')(out)
    out = Reshape((9, 16, 24))(out) 
    out = Conv2DTranspose(256, kernel_size=3, strides=1, activation='elu', padding='same')(out)
    out = Conv2DTranspose(256, kernel_size=5, strides=2, activation='elu', padding='same', name='decoder_6')(out)
    out = Conv2DTranspose(128, kernel_size=5, strides=2, activation='elu', padding='same', name='decoder_5')(out)
    out = Conv2DTranspose(64, kernel_size=5, strides=2, activation='elu', padding='same', name='decoder_4')(out)
    out = Conv2DTranspose(32, kernel_size=5, strides=2, activation='elu', padding='same', name='decoder_3')(out)
    out = Conv2DTranspose(16, kernel_size=3, activation='elu', padding='same', name='decoder_2')(out)
    out = Conv2D(3, kernel_size=1, activation='tanh', padding='same', name='decoder_1')(out)
    m = Model(inputs=input_tensor, outputs=out)
    m.compile(loss=mean_squared_error, optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                                                      epsilon=None, decay=decay, amsgrad=False))
    m.load_weights(weights_file, by_name=True, reshape=True, skip_mismatch=True)

and following error occurs:

    ValueError: Weights must be of equal size to apply a reshape operation.

When we call load_weights with skip_mismatch=True parameter, it should just skip incompatible layers. 
If fails however which is confusing.

## Keras VS TensorFlow
### Prototyping
If you really want to build a model quickly and write code concisely, then Keras is a go. The Model and the Sequential 
APIs are so powerful that they won’t even give you a sense that you are the building powerful models due to the ease in 
using them .
### Flexibility
As tensorflow is a low-level library when compared to Keras , many new functions can be implemented in a better way in 
tensorflow than in Keras for example , any activation fucntion etc… And also the fine-tuning and tweaking of the model 
is very flexible in tensorflow than in Keras due to much more parameters being available.
### Training Time and Efficiency
If models built by these two frameworks were trained on the same dataset , we see that Keras takes loner time to train 
than tensorflow . May be we cannot compare steps with epochs , the prediction accuracy is very similar and we can depict
that Keras trains slower than Tensorflow. This is a main drawback of Keras. This makes sense because of TensorFlow being
 a low level library.
 
 ## Demo Application
 I made a simple classifier application utilizing Keras library. I import Keras from TensorFlow backend to build a 
 convolutional neural network which predict the category of the input images. I got images of cars and trucks in two 
 different folders. I process and label the input images and split them into training and testing set using 
 train_test_split function. I built the model using Flatten and Dense function and used AdamOptimizder for 
 fully-connected layers. I successfully get the prediction result and the accuracy is around 80%.
 [Demo Results]()

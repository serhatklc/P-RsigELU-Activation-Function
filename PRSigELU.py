#Example of P+RSigELU activation function over MNIST dataset
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation,BatchNormalization
from keras import initializers
from keras.engine.base_layer import InputSpec
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.engine.base_layer import Layer
import tensorflow as tf
from keras.activations import sigmoid,elu
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import regularizers
import math

learning_rate = 0.01
epochs = 50
batch_size = 32
num_classes = 10
weight_decay = 0.0005
class PPRsigELU(Layer):
    """Serhat KILIÇARSLAN Parametric RSigELU version of a Rectified Linear Unit.
           https://www.sciencedirect.com/science/article/abs/pii/S0957417421002463)
    """

    def __init__(self, alpha_initializer ='GlorotNormal',beta_initializer ='GlorotNormal', **kwargs):
        super(PPRsigELU, self).__init__(**kwargs)
        self.supports_masking = True
        
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.beta_initializer = initializers.get(beta_initializer)

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=[1], name='alpha', 
                                    initializer=self.alpha_initializer,
                                     )
        self.beta = self.add_weight(shape=[1], name='beta', 
                                    initializer=self.beta_initializer,
                                     )
        self.input_spec = InputSpec(ndim=len(input_shape))
        self.built = True

    def call(self, inputs): #get_output
        alpha = 0.5
        beta = 0.5
        #return tf.maximum(0.0,inputs) #RELU
        #return tf.where(inputs > 1.0,  inputs * K.sigmoid(inputs)*self.alpha +inputs, tf.where(inputs <= 0.0, self.alpha * (K.exp(inputs) - 1.0), inputs)) #This section presents single parametric RSigELU activation function. 
        #return tf.where(inputs > 1.0,  inputs * K.sigmoid(inputs)*self.alpha +inputs, tf.where(inputs <= 0.0, self.beta * (K.exp(inputs) - 1.0), inputs)) #This section presents double parametric RSigELU activation function. 
        #return tf.where(inputs > 1.0,  inputs * K.sigmoid(inputs)*alpha +inputs, tf.where(inputs <= 0.0, alpha * (K.exp(inputs) - 1.0), inputs)) #This section presents single  RSigELU activation function. 
        return tf.where(inputs > 1.0,  inputs * K.sigmoid(inputs)*alpha +inputs, tf.where(inputs <= 0.0, beta * (K.exp(inputs) - 1.0), inputs)) #This section presents double  RSigELU activation function. 
    def get_config(self):
        config = {
                  'alpha_initializer': initializers.serialize(self.alpha_initializer),
                  'beta_initializer': initializers.serialize(self.beta_initializer)}
        base_config = super(PPRsigELU, self).get_config()
        print("Alpha Degeri",self.alpha)
        print("Beta Degeri",self.beta)
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
from tensorflow.keras.utils import to_categorical

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
#Example
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same',  input_shape=x_train.shape[1:]))
model.add(Activation(PPRsigELU()))
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation(PPRsigELU()))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4)) 
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(num_classes, activation='softmax'))

"""model.summary()"""
# Adam Optimizer

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=40,
              shuffle=True)

acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test loss:', acc[0])
print('Test accuracy:', acc[1])


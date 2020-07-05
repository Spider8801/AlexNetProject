#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:46:04 2020

@author: andromeda
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from datetime import datetime
from stellar_field_utils import load_image_dataset
from random import  sample
from os.path import exists
from os import makedirs

# Stellar dataset parameters.
num_coordinates_parameters = 2 # center x,y coordinates/ can include corners

# Training parameters.
learning_rate = 1e-4
training_steps = 2000
batch_size = 128
display_step = 10

# Network parameters.
conv1_filters = 32 # number of filters for 1st conv layer.
conv2_filters = 64 # number of filters for 2nd conv layer.
fc1_units = 1024 # number of neurons for 1st fully-connected layer.


# Prepare Stellar data.
(x_train, y_train), (x_test, y_test) = load_image_dataset()
n_validation=len(x_test) #No. of validation samples 
# Convert to float32.
#x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
#x_train, x_test = x_train / 255., x_test / 255.


# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(50000).batch(batch_size).prefetch(1)

# Create TF Model.
class ConvNet(Model):
    # Set layers.
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = layers.Conv2D(96, kernel_size=11,  padding='same', activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2, strides=2,  padding= 'same')
        self.norm1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(256, kernel_size=5, padding= 'same', activation=tf.nn.relu)
        self.maxpool2 =  layers.MaxPool2D(2, strides=2,  padding= 'same')
        self.norm2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(384, kernel_size=3, padding= 'same', activation=tf.nn.relu)
        self.norm3 = layers.BatchNormalization()
        
        self.conv4 = layers.Conv2D(384, kernel_size=3,  padding= 'same', activation=tf.nn.relu)
        self.conv5 = layers.Conv2D(256, kernel_size=3, padding= 'same', activation=tf.nn.relu)
        self.maxpool3 =  layers.MaxPool2D(2, strides=2,  padding= 'same')
        self.norm4 = layers.BatchNormalization()
        
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096,  activation=tf.nn.relu)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.fc2 = layers.Dense(4096,  activation=tf.nn.relu)
        self.dropout2 = layers.Dropout(rate=0.5)
        self.fc3 = layers.Dense(1024,  activation=tf.nn.relu)
        
        # Apply Dropout (if is_training is False, dropout is not applied).
        #self.dropout = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_coordinates_parameters)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.norm1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.norm2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.norm4(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=is_training)
        x = self.fc2(x)
        x = self.dropout2(x, training=is_training)
        x = self.fc3(x)
        x = self.out(x)
        #if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            #x = tf.nn.softmax(x)
        return x

# Build neural network model.
conv_net = ConvNet()
logfile_base_dir = "logs/scalars/"
if not exists(logfile_base_dir):
    makedirs(logfile_base_dir)
train_log_dir = logfile_base_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

def mean_square_loss(y_pred, y_true):
    # Convert true coordinates to float32 
    y_true = tf.cast(y_true, tf.float32)
    tensor_shape=y_true.get_shape().as_list()
    n_samples = tensor_shape[0]
    
    return tf.reduce_sum(tf.pow(y_pred-y_true, 2)) / n_samples

# Accuracy metric.
'''
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
'''

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# Optimization process. 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = conv_net(x, is_training=True)
        # Compute loss.
        loss = mean_square_loss(pred, y)
        
    # Variables to update, i.e. trainable variables.
    trainable_variables = conv_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss
    


# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    training_loss=run_optimization(batch_x, batch_y)
    with train_summary_writer.as_default():
        tf.summary.scalar('Training Loss', training_loss, step=step)

    if step % display_step == 0:
        pred = conv_net(batch_x)
        loss = mean_square_loss(pred, batch_y)
        #acc = accuracy(pred, batch_y)
        print('\n', "step: %i, Training loss: %f" % (step, loss))

        # Test model on validation set.
        'Randomly choose some validation samples'
        randomlist = sample(range(n_validation), batch_size)
        pred = conv_net(x_test[randomlist,:])
        validation_loss = mean_square_loss(pred, y_test[randomlist,:])
        with train_summary_writer.as_default():
            tf.summary.scalar('Validation Loss', validation_loss, step=step)
        print("Validation loss: %f" % validation_loss)
        
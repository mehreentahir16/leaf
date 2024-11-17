import os

# Suppress OpenMP informational messages
os.environ['KMP_AFFINITY'] = 'none'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import tensorflow as tf

from model import Model
from utils.model_utils import batch_data


IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        inputs = tf.keras.Input(shape=(28 * 28,), name='features')
        input_layer = tf.keras.layers.Reshape((28, 28, 1))(inputs)
        conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)(input_layer)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
        conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)(pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
        pool2_flat = tf.keras.layers.Flatten()(pool2)
        dense = tf.keras.layers.Dense(units=2048, activation=tf.nn.relu)(pool2_flat)
        logits = tf.keras.layers.Dense(units=self.num_classes)(dense)

        return tf.keras.Model(inputs=inputs, outputs=logits)

    def loss_fn(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def metrics(self):
        return ['accuracy']

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def train(self, data, num_epochs, batch_size):
        for i in range(num_epochs):
            self.run_epoch(data, batch_size)
        update = self.get_params()
        
        batch_processing = len(data['y']) // batch_size
        if batch_processing == 0:
            batch_processing = len(data['y']) / batch_size

        comp = num_epochs * batch_processing * batch_size * self.flops
        return comp, update

    def run_epoch(self, data, batch_size):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            
            with tf.GradientTape(persistent=True) as tape:
                logits = self.model(input_data, training=True)
                loss = self.model.loss(target_data, logits)
                
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            epoch_loss += loss.numpy()
            correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1), target_data), tf.float32))
            epoch_accuracy += correct_predictions.numpy()
            num_batches += 1
        
        self.stored_gradients = gradients
        
        epoch_loss /= num_batches
        epoch_accuracy /= num_batches * batch_size
        
        print(f"Epoch: Loss = {epoch_loss}, Accuracy = {epoch_accuracy}") 

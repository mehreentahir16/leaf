import os

# Suppress OpenMP informational messages
os.environ['KMP_AFFINITY'] = 'none'
os.environ['OMP_NUM_THREADS'] = '1'

import tensorflow as tf

from model import Model
import numpy as np


IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        features = tf.compat.v1.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.compat.v1.placeholder(tf.int64, shape=[None], name='labels')
        input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.compat.v1.train.get_global_step())
        eval_metric_ops = tf.math.count_nonzero(tf.equal(labels, predictions["classes"]))
        gradients = tf.gradients(loss, tf.compat.v1.trainable_variables())
        return features, labels, train_op, eval_metric_ops, loss, gradients

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
    
    def hmc_sample(self, num_samples, num_burnin_steps, step_size):
        def leapfrog(position, momentum, step_size):
            grads = self.stored_gradients
            new_momentum = [m + 0.5 * step_size * g for m, g in zip(momentum, grads)]
            new_position = [p + step_size * nm for p, nm in zip(position, new_momentum)]
            new_momentum = [m + 0.5 * step_size * g for m, g in zip(new_momentum, grads)]
            return new_position, new_momentum

        with self.graph.as_default():
            initial_params = self.get_params()
            position = initial_params
            samples = []
            for step in range(num_burnin_steps + num_samples):
                momentum = [np.random.randn(*p.shape) for p in position]
                current_position = position
                current_momentum = momentum
                for _ in range(3):
                    current_position, current_momentum = leapfrog(current_position, current_momentum, step_size)
                if step >= num_burnin_steps:
                    samples.append(current_position)
                position = current_position
            samples = np.array(samples)
            mean = np.mean(samples, axis=0)
            variance = np.var(samples, axis=0)
            print("mean len in hmc method...", len(mean))
            print("variance len in hmc method", len(variance))
            return mean, variance

import tensorflow as tf
import numpy as np
from model import Model

IMAGE_SIZE = 32
NUM_CLASSES = 10

class CIFAR10Model(Model):
    def __init__(self, seed, lr, num_classes=NUM_CLASSES, optimizer=None):
        self.num_classes = num_classes
        super(CIFAR10Model, self).__init__(seed, lr, optimizer)

    def create_model(self):
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')

        # Layer 1: Conv -> BatchNorm -> ReLU -> Pool
        conv1 = tf.layers.conv2d(inputs=features, filters=32, kernel_size=[3, 3], padding="same", activation=None)
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Layer 2: Conv -> BatchNorm -> ReLU -> Pool
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=None)
        conv2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Layer 3: Conv -> BatchNorm -> ReLU -> Pool
        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=None)
        conv3 = tf.layers.batch_normalization(conv3)
        conv3 = tf.nn.relu(conv3)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        # Flatten layer
        flat = tf.layers.flatten(pool3)

        # Fully connected layer with dropout
        dense = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.5)

        # Output layer
        logits = tf.layers.dense(inputs=dropout, units=self.num_classes)

        # Loss and optimizer
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = self.optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        # Evaluation metric
        eval_metric_ops = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

        return features, labels, train_op, eval_metric_ops, loss

    def process_x(self, raw_x_batch):
        # Normalize images to range [-1, 1]
        x_data = np.array(raw_x_batch, dtype=np.float32)
        x_data = (x_data - 127.5) / 127.5
        return x_data

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)


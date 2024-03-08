"""Interfaces for ClientModel and ServerModel."""

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from baseline_constants import ACCURACY_KEY
from utils.model_utils import batch_data

class Model(ABC):

    def __init__(self, seed, lr, optimizer=None):
        self.lr = lr
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Initialize the model architecture in subclasses
        self.model = self.create_model()

        # Set the optimizer
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        else:
            self.optimizer = optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 4-tuple consisting of:
                features: A placeholder for the samples' features.
                labels: A placeholder for the samples' labels.
                train_op: A Tensorflow operation that, when run with the features and
                    the labels, trains the model.
                eval_metric_ops: A Tensorflow operation that, when run with features and labels,
                    returns the accuracy of the model.
        """
        pass

    def get_params(self):
        """
        Returns the model's parameters.
        """
        return self.model.get_weights()

    def set_params(self, model_params):
        """
        Sets the model's parameters.
        """
        self.model.set_weights(model_params)

    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        for epoch in range(num_epochs):
            self.run_epoch(data, batch_size)

        update = self.get_params()
        comp = self.calculate_comp(num_epochs, data, batch_size)
        return comp, update

    def run_epoch(self, data, batch_size):
        """
        Run training for one epoch. Implementations may vary based on the model's requirements.
        This method processes the data, organizes it into batches, and applies the training step.
        """
        dataset = tf.data.Dataset.from_tensor_slices((self.process_x(data['x']), self.process_y(data['y'])))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

        for batch_x, batch_y in dataset:
            self.train_step(batch_x, batch_y)

    def train_step(self, x_batch, y_batch):
        """
        Executes a single training step.
        """
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            loss = self.compute_loss(predictions, y_batch)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def test(self, data):
        """
        Tests the model's performance on the provided data.
        """
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        print("x_vecs shape", x_vecs.shape)
        print("labels shape", labels.shape)
        eval_result = self.model.evaluate(x_vecs, labels, verbose=0)
        acc = eval_result[1]  # Assuming accuracy is the second output
        loss = eval_result[0]
        return {ACCURACY_KEY: acc, 'loss': loss}

    def compute_loss(self, predictions, labels):
        """
        Computes the loss using the model's predictions and the true labels.
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predictions))

    def calculate_comp(self, num_epochs, data, batch_size):
        """
        Placeholder method for calculating the computational cost of training.
        Adapt this method based on your model's specifics.
        """
        return num_epochs * len(data['y']) / batch_size

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass

class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        # This property was intended to represent the number of parameters or memory size.
        # Since TensorFlow 2.x doesn't directly provide a 'size' attribute, we'll calculate it as follows:
        return sum(np.prod(v.shape) for v in self.model.get_weights())

    @property
    def cur_model(self):
        # Direct access to the model instance
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        # In TensorFlow 2.x, we directly get weights as numpy arrays and set them in client models.
        model_weights = self.model.get_weights()
        for client in clients:
            client.model.set_weights(model_weights)

    def save(self, path='model_checkpoint'):
        # TensorFlow 2.x uses the Keras API for saving the whole model or just the weights.
        # This example saves the entire model for simplicity.
        self.model.save(path)

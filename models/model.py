"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from baseline_constants import ACCURACY_KEY

from utils.model_utils import batch_data
from utils.tf_utils import graph_size


class Model(ABC):

    def __init__(self, seed, lr, optimizer=None):
        self.lr = lr
        self.seed = seed
        self._optimizer = optimizer

        # No more graph definition in TensorFlow 2.x
        self.model = self.create_model()
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn(), metrics=self.metrics())

        np.random.seed(self.seed)
        self.stored_gradients = None

        # Calculate FLOPs and model size
        self.flops = self.calculate_flops()
        self.size = self.calculate_model_size()

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task."""
        pass

    @abstractmethod
    def loss_fn(self):
        """Defines the loss function."""
        pass

    @abstractmethod
    def metrics(self):
        """Defines the metrics for evaluation."""
        pass

    def set_params(self, model_params):
        for variable, value in zip(self.model.trainable_variables, model_params):
            variable.assign(value)

    def get_params(self):
        return self.model.trainable_variables

    def train(self, data, num_epochs, batch_size):
        # print(f"Current learning rate: {self.optimizer.learning_rate.numpy()}")
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
            
            with tf.GradientTape() as tape:
                logits = self.model(input_data, training=True)
                loss = self.model.compiled_loss(target_data, logits)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            epoch_loss += loss.numpy()
            correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1), target_data), tf.float32))
            epoch_accuracy += correct_predictions.numpy()
            num_batches += 1

            # # Logging details for each batch
            # print(f"Batch {num_batches}: Loss = {loss.numpy()}, Accuracy = {correct_predictions.numpy() / len(target_data)}")
        
        self.stored_gradients = gradients
        
        # Averaging over all batches
        epoch_loss /= num_batches
        epoch_accuracy /= num_batches * batch_size
        
        # Logging epoch details
        print(f"Epoch: Loss = {epoch_loss}, Accuracy = {epoch_accuracy}") 

    def test(self, data):
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        evaluation = self.model.evaluate(x_vecs, labels, return_dict=True)
        return {ACCURACY_KEY: evaluation['accuracy']*100, 'loss': evaluation['loss']}

    def close(self):
        pass

    @abstractmethod
    def process_x(self, raw_x_batch):
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        pass

    @staticmethod
    @tf.function(reduce_retracing=True)
    def model_forward_pass(inputs, model):
        return model(inputs)

    def calculate_flops(self):
        concrete_func = self.model_forward_pass.get_concrete_function(
            tf.TensorSpec([1] + list(self.model.input.shape[1:]), self.model.input.dtype), self.model)

        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops if flops else 0
    
    def calculate_model_size(self):
        total_parameters = 0
        for variable in self.model.trainable_variables:
            shape = variable.shape
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            dtype_size = tf.dtypes.as_dtype(variable.dtype).size  # Get the size of the data type in bytes
            total_parameters += variable_parameters * dtype_size
        return total_parameters


class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        return graph_size(self.model)

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        var_vals = {v.name: v.numpy() for v in self.model.get_params()}
        for c in clients:
            c.model.set_params(var_vals.values())

    def save(self, path='checkpoints/model.ckpt'):
        self.model.save_weights(path)

    def close(self):
        self.model.close()

import os

# Suppress OpenMP informational messages
os.environ['KMP_AFFINITY'] = 'none'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from model import Model


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

    def target_log_prob_fn(self, cloned_model, *weights):
        for variable, weight in zip(cloned_model.trainable_variables, weights):
            variable.assign(weight)
        log_prob = -tf.reduce_mean(
            [tf.reduce_sum(g * w) for g, w in zip(self.stored_gradients, weights)])
        return log_prob

    @tf.function
    def run_chain(self, initial_state, num_samples, num_burnin_steps, kernel):
        with tf.device('/GPU:0'):
            return tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin_steps,
                current_state=initial_state,
                kernel=kernel,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
            )

    def nuts_sample(self, num_samples=10, num_burnin_steps=5, step_size=0.001):
        # Clone the model
        cloned_model = tf.keras.models.clone_model(self.model)
        cloned_model.set_weights(self.model.get_weights())

        initial_state = [tf.identity(var) for var in cloned_model.trainable_variables]

        # Use NUTS kernel
        nuts_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=lambda *args: self.target_log_prob_fn(cloned_model, *args),
            step_size=step_size,
            max_tree_depth=5,
            parallel_iterations=10)

        adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=nuts_kernel,
            num_adaptation_steps=int(num_burnin_steps * 0.8),
            target_accept_prob=0.75)

        samples, kernel_results = self.run_chain(
            initial_state=initial_state,
            num_samples=num_samples,
            num_burnin_steps=num_burnin_steps,
            kernel=adaptive_kernel
        )

        samples = [sample.numpy() for sample in samples]
        acceptance_rate = np.mean(kernel_results)

        print("acceptance rate", acceptance_rate)

        mean = [np.mean(sample, axis=0) for sample in samples]
        variance = [np.var(sample, axis=0) for sample in samples]

        # Debugging output
        for i, (m, v) in enumerate(zip(mean, variance)):
            print(f"Layer {i}: mean shape: {m.shape}, variance shape: {v.shape}")

        return mean, variance
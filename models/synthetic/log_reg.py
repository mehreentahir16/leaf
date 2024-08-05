import numpy as np
import tensorflow as tf
from model import Model
from utils.model_utils import batch_data

class ClientModel(Model):
    def __init__(self, seed, lr, num_classes, input_dim):
        self.num_classes = num_classes
        self.input_dim = input_dim
        super(ClientModel, self).__init__(seed, lr)
        self.mean = [tf.Variable(tf.zeros_like(v)) for v in self.model.trainable_variables]
        self.log_variance = [tf.Variable(tf.zeros_like(v)) for v in self.model.trainable_variables]
        self.var_optimizer_mean = tf.keras.optimizers.Adam()
        self.var_optimizer_log_variance = tf.keras.optimizers.Adam()

    def create_model(self):
        inputs = tf.keras.Input(shape=(self.input_dim,), name='features')
        logits = tf.keras.layers.Dense(self.num_classes)(inputs)  # No activation function
        model = tf.keras.Model(inputs=inputs, outputs=logits)
        return model

    def loss_fn(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def metrics(self):
        return ['accuracy']

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def train(self, data, num_epochs, batch_size):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0

        for epoch in range(num_epochs):
            for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
                input_data = self.process_x(batched_x)
                target_data = self.process_y(batched_y)

                with tf.GradientTape(persistent=True) as tape:
                    logits = self.model(input_data, training=True)
                    loss = self.model.compiled_loss(target_data, logits)
                    
                    # Add KL divergence to the loss
                    kl_divergence = 0
                    for mean, log_var in zip(self.mean, self.log_variance):
                        kl_divergence += tf.reduce_sum(-0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var)))
                    loss += kl_divergence
                    
                    # # Proximal term for FedProx
                    # if global_params is not None:
                    #     prox_term = 0
                    #     for w, w_global in zip(self.model.trainable_variables, global_params):
                    #         prox_term += tf.reduce_sum(tf.square(w - w_global))
                    #     loss += (mu / 2) * prox_term

                gradients = tape.gradient(loss, self.model.trainable_variables)
                mean_gradients = tape.gradient(loss, self.mean)
                log_variance_gradients = tape.gradient(loss, self.log_variance)
                
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                self.var_optimizer_mean.apply_gradients(zip(mean_gradients, self.mean))
                self.var_optimizer_log_variance.apply_gradients(zip(log_variance_gradients, self.log_variance))
                
                epoch_loss += loss.numpy()
                correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1), target_data), tf.float32))
                epoch_accuracy += correct_predictions.numpy()
                num_batches += 1

            epoch_loss /= num_batches
            epoch_accuracy /= num_batches * batch_size
            print(f"Epoch: {epoch + 1}, Loss = {epoch_loss}, Accuracy = {epoch_accuracy}")

        elbo = self.calculate_elbo(data, batch_size)
        return num_batches * batch_size, self.get_params(), elbo, [v.numpy() for v in self.log_variance]

    def calculate_elbo(self, data, batch_size):
        elbo_total = 0
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            elbo_total += self.elbo(input_data, target_data).numpy()
        return elbo_total / (len(data['y']) / batch_size)

    def elbo(self, input_data, target_data):
        logits = self.model(input_data, training=False)
        loss = self.model.compiled_loss(target_data, logits)
        
        kl_divergence = 0
        for mean, log_var in zip(self.mean, self.log_variance):
            kl_divergence += tf.reduce_sum(-0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var)))
        
        elbo = loss + kl_divergence
        return elbo

    def test(self, data):
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])

        logits = self.model(x_vecs, training=False)
        loss = self.model.compiled_loss(labels, logits)
        tot_acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.float32)).numpy()

        acc = float(tot_acc) / len(x_vecs)
        return {'accuracy': acc, 'loss': loss.numpy()}

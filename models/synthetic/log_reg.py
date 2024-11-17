import numpy as np
import tensorflow as tf
from model import Model
from utils.model_utils import batch_data

class ClientModel(Model):
    def __init__(self, seed, lr, num_classes, input_dim):
        self.num_classes = num_classes
        self.input_dim = input_dim
        super(ClientModel, self).__init__(seed, lr)

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
                    loss = self.model.loss(target_data, logits)
                            
                gradients = tape.gradient(loss, self.model.trainable_variables)
                
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                epoch_loss += loss.numpy()
                correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1), target_data), tf.float32))
                epoch_accuracy += correct_predictions.numpy()
                num_batches += 1

            epoch_loss /= num_batches
            epoch_accuracy /= num_batches * batch_size
            print(f"Epoch: {epoch + 1}, Loss = {epoch_loss}, Accuracy = {epoch_accuracy}")


        return num_batches * batch_size, self.get_params()

    def test(self, data):
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])

        logits = self.model(x_vecs, training=False)
        loss = self.model.loss(labels, logits)
        tot_acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.float32)).numpy()

        acc = float(tot_acc) / len(x_vecs)
        return {'accuracy': acc, 'loss': loss.numpy()}

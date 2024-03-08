import numpy as np
import tensorflow as tf

from model import Model
from utils.language_utils import letter_to_vec, word_to_indices

class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        super(ClientModel, self).__init__(seed, lr)

        self.model.compile(optimizer=self.optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    def create_model(self):
        inputs = tf.keras.Input(shape=(self.seq_len,), dtype=tf.int32)
        embedding_layer = tf.keras.layers.Embedding(input_dim=self.num_classes, output_dim=8)
        embedding = embedding_layer(inputs)

        lstm = tf.keras.layers.LSTM(self.n_hidden, return_sequences=True)
        lstm2 = tf.keras.layers.LSTM(self.n_hidden, return_sequences=False)
        lstm_output = lstm(embedding)
        lstm2_output = lstm2(lstm_output)

        pred = tf.keras.layers.Dense(self.num_classes, activation='softmax')(lstm2_output)

        model = tf.keras.Model(inputs=inputs, outputs=pred)

        return model

    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            logits = self.model(x_batch, training=True)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(y_batch, dtype=tf.float32)))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return loss, accuracy

    def process_x(self, raw_x_batch):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        return np.array(y_batch)
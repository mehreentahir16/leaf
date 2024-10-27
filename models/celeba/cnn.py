import numpy as np
import os
import tensorflow as tf
from PIL import Image
from model import Model
from utils.model_utils import batch_data

IMAGE_SIZE = 84
IMAGES_DIR = os.path.join('..', 'data', 'celeba', 'data', 'raw', 'img_align_celeba')

class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)
        self.var_optimizer_mean = tf.keras.optimizers.Adam(learning_rate=lr)
        self.var_optimizer_log_variance = tf.keras.optimizers.Adam(learning_rate=lr)

    def create_model(self):
        input_ph = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        out = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(input_ph)
        out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(out)
        
        out = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(out)
        out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(out)
        
        out = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(out)
        out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(out)
        
        out = tf.keras.layers.Flatten()(out)
        logits = tf.keras.layers.Dense(1, activation=None)(out)  # Output a single logit for binary classification
        model = tf.keras.Model(inputs=input_ph, outputs=logits)
        
        # Initialize mean and log_variance as trainable variables
        self.mean = [tf.Variable(tf.random.normal(shape=var.shape), trainable=True) for var in model.trainable_variables]
        self.log_variance = [tf.Variable(tf.zeros(shape=var.shape), trainable=True) for var in model.trainable_variables]

        model.compile(optimizer=self.optimizer, loss=self.loss_fn(), metrics=self.metrics())
        return model

    def loss_fn(self):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def metrics(self):
        return ['accuracy']

    def process_x(self, raw_x_batch):
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(x_batch) / 255.0  # Normalize input images
        return x_batch

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def _load_image(self, img_name):
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        return np.array(img)

    def elbo(self, input_data, target_data):
        logits = self.model(input_data, training=True)
        log_likelihood = -self.loss_fn()(target_data, logits)
        
        kl_divergence = 0
        for mean, log_var in zip(self.mean, self.log_variance):
            kl_divergence += tf.reduce_sum(-0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var)))
        
        return log_likelihood - kl_divergence

    def train(self, data, num_epochs, batch_size, global_params):
        for i in range(num_epochs):
            self.run_epoch(data, batch_size, global_params)
        update = self.get_params()
        elbo = self.calculate_elbo(data, batch_size)
        variance = [tf.reduce_mean(tf.exp(log_var)).numpy() for log_var in self.log_variance]
        
        batch_processing = len(data['y']) // batch_size
        if batch_processing == 0:
            batch_processing = len(data['y']) / batch_size

        comp = num_epochs * batch_processing * batch_size * self.flops
        return comp, update, elbo, variance

    def run_epoch(self, data, batch_size, global_params=None):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        mu=0.01
        
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            
            with tf.GradientTape(persistent=True) as tape:
                logits = self.model(input_data, training=True)
                loss = self.model.compiled_loss(target_data, logits)

                # Add proximal term
                if global_params is not None:
                    prox_term = tf.add_n([tf.reduce_sum(tf.square(var - gvar)) for var, gvar in zip(self.model.trainable_variables, global_params)])
                    loss += (mu / 2) * prox_term
                
                # # Add KL divergence to the loss
                # kl_divergence = 0
                # for mean, log_var in zip(self.mean, self.log_variance):
                #     kl_divergence += tf.reduce_sum(-0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var)))
                
                # loss += kl_divergence
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            mean_gradients = tape.gradient(loss, self.mean)
            log_variance_gradients = tape.gradient(loss, self.log_variance)
            
            if gradients and any(g is not None for g in gradients):
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if mean_gradients and any(g is not None for g in mean_gradients):
                self.var_optimizer_mean.apply_gradients(zip(mean_gradients, self.mean))
            if log_variance_gradients and any(g is not None for g in log_variance_gradients):
                self.var_optimizer_log_variance.apply_gradients(zip(log_variance_gradients, self.log_variance))
            
            epoch_loss += loss.numpy()
            correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.round(tf.nn.sigmoid(logits)), target_data), tf.float32))
            epoch_accuracy += correct_predictions.numpy()
            num_batches += 1
        
        self.stored_gradients = gradients
        
        epoch_loss /= num_batches
        epoch_accuracy /= num_batches * batch_size
        
        print(f"Epoch: Loss = {epoch_loss}, Accuracy = {epoch_accuracy}")

    def calculate_elbo(self, data, batch_size):
        elbo_total = 0
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            elbo_total += self.elbo(input_data, target_data).numpy()
        return elbo_total / (len(data['y']) / batch_size)
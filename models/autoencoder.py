import tensorflow as tf

class LayerWiseAutoencoder:
    def __init__(self, layer_dims, bottleneck_dims, learning_rate=0.001):
        """
        :param layer_dims: List of input dimensions for each layer.
        :param bottleneck_dims: List of bottleneck dimensions for each layer.
        :param learning_rate: Learning rate for training.
        """
        self.layer_dims = layer_dims
        self.bottleneck_dims = bottleneck_dims
        self.learning_rate = learning_rate
        self._build_model()
    
    def _build_model(self):
        self.inputs = []
        self.reconstructed_outputs = []
        self.reconstruction_losses = []
        self.total_loss = 0
        self.encoder_vars = []
        self.decoder_vars = []
        
        for idx, (layer_dim, bottleneck_dim) in enumerate(zip(self.layer_dims, self.bottleneck_dims)):
            with tf.variable_scope(f'autoencoder_layer_{idx}'):
                layer_input = tf.placeholder(tf.float32, [None, layer_dim], name=f'input_layer_{idx}')
                self.inputs.append(layer_input)
                
                # Encoder
                W_enc = tf.get_variable('W_enc', shape=[layer_dim, bottleneck_dim],
                                        initializer=tf.contrib.layers.xavier_initializer())
                b_enc = tf.get_variable('b_enc', shape=[bottleneck_dim],
                                        initializer=tf.zeros_initializer())
                encoded = tf.nn.relu(tf.matmul(layer_input, W_enc) + b_enc)
                self.encoder_vars.extend([W_enc, b_enc])
                
                # Decoder with residual connections
                W_dec = tf.get_variable('W_dec', shape=[bottleneck_dim, layer_dim],
                                        initializer=tf.contrib.layers.xavier_initializer())
                b_dec = tf.get_variable('b_dec', shape=[layer_dim],
                                        initializer=tf.zeros_initializer())
                decoded = tf.matmul(encoded, W_dec) + b_dec
                self.decoder_vars.extend([W_dec, b_dec])
                
                # Reconstruction with skip connection
                decoded = decoded + layer_input  # Residual connection
                
                # Reconstruction loss
                reconstruction_loss = tf.reduce_mean(tf.square(decoded - layer_input))
                self.reconstruction_losses.append(reconstruction_loss)
                self.total_loss += reconstruction_loss
                
                self.reconstructed_outputs.append(decoded)
        
        # Regularization
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, self.encoder_vars + self.decoder_vars)
        self.total_loss += regularization_penalty
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

    def train(self, sess, X_train, num_epochs=50, batch_size=64):
        num_samples = X_train[0].shape[0]
        num_batches = max(1, num_samples // batch_size)
        for epoch in range(num_epochs):
            avg_loss = 0
            for batch_idx in range(num_batches):
                feed_dict = {}
                for i, layer_data in enumerate(X_train):
                    batch_data = layer_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    feed_dict[self.inputs[i]] = batch_data
                _, loss = sess.run([self.optimizer, self.total_loss], feed_dict=feed_dict)
                avg_loss += loss / num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {avg_loss:.6f}')

    def compute_layer_errors(self, sess, X_data):
        """Compute reconstruction errors for each layer per sample."""
        layer_errors = []
        for i, layer_data in enumerate(X_data):
            feed_dict = {self.inputs[i]: layer_data}
            # Compute per-sample reconstruction errors for this layer
            errors = sess.run(tf.reduce_mean(tf.square(self.reconstructed_outputs[i] - self.inputs[i]), axis=1), feed_dict=feed_dict)
            layer_errors.append(errors)
        return layer_errors

from sklearn.model_selection import KFold
import numpy as np

# def cross_validate_autoencoder(X_data, bottleneck_dim, learning_rate, num_epochs, batch_size, K=5):
#     kf = KFold(n_splits=K, shuffle=True, random_state=42)
#     fold = 1
#     validation_losses = []
    
#     for train_index, val_index in kf.split(X_data):
#         print(f'Fold {fold}')
#         X_train, X_val = X_data[train_index], X_data[val_index]
        
#         # Create a new graph for each fold
#         graph = tf.Graph()
#         with graph.as_default():
#             autoencoder = Autoencoder(input_dim=X_data.shape[1], bottleneck_dim=bottleneck_dim, learning_rate=learning_rate)
            
#             with tf.Session(graph=graph) as sess:
#                 sess.run(tf.global_variables_initializer())
#                 autoencoder.train(sess, X_train, num_epochs=num_epochs, batch_size=batch_size)
#                 # Compute validation loss
#                 val_loss = sess.run(autoencoder.loss, feed_dict={autoencoder.X: X_val})
#                 print(f'Validation Loss for fold {fold}: {val_loss:.6f}')
#                 validation_losses.append(val_loss)
        
#         fold += 1
    
#     avg_val_loss = np.mean(validation_losses)
#     print(f'Average Validation Loss: {avg_val_loss:.6f}')
#     return avg_val_loss
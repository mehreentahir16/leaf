import tensorflow as tf

class Autoencoder:
    def __init__(self, input_dim, bottleneck_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.learning_rate = learning_rate
        self._build_model()
    
    def _build_model(self):
        # Input placeholder
        self.X = tf.placeholder(tf.float32, [None, self.input_dim], name='input')
        
        # Encoder
        with tf.variable_scope('encoder'):
            W_enc1 = tf.Variable(tf.truncated_normal([self.input_dim, 128], stddev=0.1), name='W_enc1')
            b_enc1 = tf.Variable(tf.zeros([128]), name='b_enc1')
            enc_layer1 = tf.nn.relu(tf.matmul(self.X, W_enc1) + b_enc1)
            # enc_layer1 = tf.compat.v1.layers.batch_normalization(enc_layer1)
            
            # Bottleneck layer
            W_bottleneck = tf.Variable(tf.truncated_normal([128, self.bottleneck_dim], stddev=0.1), name='W_bottleneck')
            b_bottleneck = tf.Variable(tf.zeros([self.bottleneck_dim]), name='b_bottleneck')
            self.encoded = tf.nn.relu(tf.matmul(enc_layer1, W_bottleneck) + b_bottleneck)
            # self.encoded = tf.compat.v1.layers.batch_normalization(self.encoded)
        
        # Decoder
        with tf.variable_scope('decoder'):
            W_dec1 = tf.Variable(tf.truncated_normal([self.bottleneck_dim, 128], stddev=0.1), name='W_dec1')
            b_dec1 = tf.Variable(tf.zeros([128]), name='b_dec1')
            dec_layer1 = tf.nn.relu(tf.matmul(self.encoded, W_dec1) + b_dec1)
            # dec_layer1 = tf.compat.v1.layers.batch_normalization(dec_layer1)
            
            # Output layer
            W_dec_out = tf.Variable(tf.truncated_normal([128, self.input_dim], stddev=0.1), name='W_dec_out')
            b_dec_out = tf.Variable(tf.zeros([self.input_dim]), name='b_dec_out')
            self.decoded = tf.matmul(dec_layer1, W_dec_out) + b_dec_out  # Linear output
            
        # Loss and optimizer
        with tf.variable_scope('loss'):
            reconstruction_loss = tf.reduce_mean(tf.square(self.decoded - self.X))
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
            regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, tf.trainable_variables())
            self.loss = reconstruction_loss + regularization_penalty
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
    def train(self, sess, X_train, num_epochs=50, batch_size=64):
        num_samples = X_train.shape[0]
        num_batches = num_samples // batch_size
        for epoch in range(num_epochs):
            avg_loss = 0
            for i in range(num_batches):
                batch_x = X_train[i * batch_size : (i + 1) * batch_size]
                feed_dict = {self.X: batch_x}
                _, l = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                avg_loss += l / num_batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    def compute_reconstruction_error(self, sess, X_data):
        feed_dict = {self.X: X_data}
        # Return per-sample reconstruction errors
        return sess.run(tf.reduce_mean(tf.square(self.decoded - self.X), axis=1), feed_dict=feed_dict)

class VariationalAutoencoder:
    def __init__(self, input_dim, latent_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self._build_model()
        
    def _build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        
        # Encoder
        with tf.variable_scope('encoder'):
            h_enc = tf.layers.dense(self.X, 128, activation=tf.nn.relu)
            self.mu = tf.layers.dense(h_enc, self.latent_dim)
            self.log_var = tf.layers.dense(h_enc, self.latent_dim)
            
            # Reparameterization trick
            eps = tf.random_normal(tf.shape(self.mu))
            self.z = self.mu + tf.exp(0.5 * self.log_var) * eps
        
        # Decoder
        with tf.variable_scope('decoder'):
            h_dec = tf.layers.dense(self.z, 128, activation=tf.nn.relu)
            self.decoded = tf.layers.dense(h_dec, self.input_dim)
        
        # Loss
        with tf.variable_scope('loss'):
            # Reconstruction loss (could use binary_crossentropy if data is normalized between 0 and 1)
            recon_loss = tf.reduce_mean(tf.square(self.decoded - self.X))
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(1 + self.log_var - tf.square(self.mu) - tf.exp(self.log_var))
            self.loss = recon_loss + kl_loss
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 10000
batch_size = 128

display_step = 1000
examples_to_show = 3

# Network Parameters
num_hidden_1 = 16 # with 16 neurons 
num_hidden_2 = 8 # with 16 neurons 
num_hidden_3 = 4 # with 16 neurons 
num_input = 784 # MNIST data input (img shape: 28*28)





# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    'encoder_h2': tf.Variable(tf.random_normal([num_input, num_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_input])),
    'encoder_h3': tf.Variable(tf.random_normal([num_input, num_hidden_3])),
    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_3, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b1': tf.Variable(tf.random_normal([num_input])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    'decoder_b3': tf.Variable(tf.random_normal([num_input])),

}

# Building 3 encoders and decoders with different number of neruons
def encoder1(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))   
    return layer_1


def decoder1(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    return layer_1

def encoder2(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h2']),
                                   biases['encoder_b2']))   
    return layer_1
def decoder2(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h2']),
                                   biases['decoder_b2'])) 
    return layer_1

def encoder3(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h3']),
                                   biases['encoder_b3']))   
    return layer_1
def decoder3(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h3']),
                                   biases['decoder_b3']))
    
    return layer_1
# Construct model
encoder_op1 = encoder1(X)
decoder_op1 = decoder1(encoder_op1)
encoder_op2 = encoder2(X)
decoder_op2 = decoder2(encoder_op2)
encoder_op3 = encoder3(X)
decoder_op3 = decoder3(encoder_op3)
# Prediction
y_pred1 = decoder_op1
y_pred2 = decoder_op2
y_pred3 = decoder_op3

# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss1 = tf.reduce_mean(tf.pow(y_true - y_pred1, 2))
loss2 = tf.reduce_mean(tf.pow(y_true - y_pred2, 2))
loss3 = tf.reduce_mean(tf.pow(y_true - y_pred3, 2))

optimizer1 = tf.train.AdamOptimizer(learning_rate).minimize(loss1)
optimizer2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2)
optimizer3 = tf.train.AdamOptimizer(learning_rate).minimize(loss3)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        
        _, l1 = sess.run([optimizer1, loss1], feed_dict={X: batch_x})
        _, l2 = sess.run([optimizer2, loss2], feed_dict={X: batch_x})
        _, l3 = sess.run([optimizer3, loss3], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l1))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon1 = np.empty((28 * n, 28 * n))
    canvas_recon2 = np.empty((28 * n, 28 * n))
    canvas_recon3 = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g1 = sess.run(decoder_op1, feed_dict={X: batch_x})
        g2 = sess.run(decoder_op2, feed_dict={X: batch_x})
        g3 = sess.run(decoder_op3, feed_dict={X: batch_x})
        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon1[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g1[j].reshape([28, 28])
            canvas_recon2[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g2[j].reshape([28, 28])
            canvas_recon3[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g3[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(3, 3))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images, hidden neurons = 16")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon1, origin="upper", cmap="gray")
    plt.show()
    
    print("Reconstructed Images, hidden neurons = 8")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon2, origin="upper", cmap="gray")
    plt.show()
    
    print("Reconstructed Images, hidden neurons = 4")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon3, origin="upper", cmap="gray")
    plt.show()
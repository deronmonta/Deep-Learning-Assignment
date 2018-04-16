# An undercomplete autoencoder on MNIST dataset
from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 500  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
learning_rate = 0.001        # Learning rate

#Part 1 Reshape to 32 by 32
def resize_batch(imgs):
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs


def autoencoder(inputs):
    #32*32 -> 16*16*32 -> 8*8*16
    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net

# read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size

ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))  # input to the network (MNIST images)
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img, batch_label = mnist.train.next_batch(batch_size)  # read a batch
            batch_img = batch_img.reshape((-1, 28, 28, 1))               # reshape each sample to an (28, 28) image
            batch_img = resize_batch(batch_img)                          # reshape the images to (32, 32)
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
            #print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
        if ep == 0:
            batch_img1, batch_label1 = mnist.test.next_batch(10)
            batch_img1 = resize_batch(batch_img1)
            recon_img1 = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img1})[0]
        elif ep == 2:
            batch_img2, batch_label2 = mnist.test.next_batch(10)
            batch_img2 = resize_batch(batch_img2)
            recon_img2 = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img2})[0]
        elif ep == 4:
            batch_img3, batch_label3 = mnist.test.next_batch(10)
            batch_img3 = resize_batch(batch_img3)
            recon_img3 = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img3})[0]   
    
    plt.figure()
    print('Reconstructed Images at Epoch 0 ')
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(recon_img1[i, ..., 0], cmap='gray')
    plt.show()          
    
    plt.figure()
    print('Reconstructed Images at Epoch: 3')
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(recon_img2[i, ..., 0], cmap='gray')
    plt.show()          
    
    plt.figure()
    print('Reconstructed Images at Epoch: 5')
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(recon_img3[i, ..., 0], cmap='gray')
    plt.show()          
    
    
    plt.figure()
    print('Input Images')
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(batch_img[i, ..., 0], cmap='gray')
    plt.show()
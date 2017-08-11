# -*- coding:utf-8 -*-

import cmodel
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

print 'train_examples', mnist.train.num_examples
print 'test_examples', mnist.test.num_examples

# Parameters
learning_rate = 1e-4
k_dg = 1
batch_size = 256
training_epochs = 500
data_size = 28
noise_size = 100
show_step = 10
plot_step = mnist.train.num_examples / batch_size * 200

# placeholder
z = tf.placeholder(tf.float32, [None, noise_size])
x = tf.placeholder(tf.float32, [None, data_size, data_size, 1])
keep_prob_d = tf.placeholder(tf.float32)

# structure: [[channels, height, width],
# [out_channels, filter_height, filter_width, stride_width, stride_height, out_height, out_width]...]
generated_x = cmodel.generative_model(
	[z], batch_size, [[12, 7, 7], [6, 5, 5, 2, 2, 14, 14], [1, 5, 5, 2, 2, 28, 28]])
# structure: [[out_channels, filter_height, filter_width, stride_height, stride_width]...fc_layers]
# keep_prob_d is only effective in fc_layers except the discriminate layer
prob_x, prob_z = cmodel.discriminative_model(
	[x, generated_x], [[6, 5, 5, 2, 2], [12, 5, 5, 2, 2], [None, 1]], keep_prob_d)
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]
print d_vars
print g_vars
score, x_accuracy, z_accuracy, train_D = cmodel.maximize_discriminative_model(
	prob_x, prob_z, learning_rate, var_list=d_vars)
cost, zg_accuracy, train_G = cmodel.minimize_generative_model(prob_z, learning_rate, var_list=g_vars)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	step = 0
	while True:
		for _ in xrange(k_dg):
			step += 1
			noise_z = cmodel.sample_noise(batch_size, noise_size)
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			batch_x = cmodel.normalize(batch_x.reshape(-1, data_size, data_size, 1))
			d_result = sess.run([score, x_accuracy, z_accuracy, train_D], feed_dict={
				z: noise_z, x: batch_x, keep_prob_d: 0.8})

		noise_z = cmodel.sample_noise(batch_size, noise_size)
		g_result = sess.run([cost, zg_accuracy, train_G], feed_dict={
			z: noise_z, x: batch_x, keep_prob_d: 0.8})

		if step % show_step == 0:
			print 'Epoch %d, Process %f' % (step * batch_size / mnist.train.num_examples, step * batch_size %
											mnist.train.num_examples / float(mnist.train.num_examples))
			print 'Discriminative Model Score %f, Accuracy %f %f' % (d_result[0].mean(), d_result[1], d_result[2])
			print 'Generated Model Cost %f, Accuracy %f' % (g_result[0].mean(), g_result[1])

		if step % plot_step == 0:
			noise_z = cmodel.sample_noise(batch_size, noise_size)
			g_result = sess.run([generated_x], feed_dict={
				z: noise_z, x: batch_x, keep_prob_d: 1.0})

			for i in xrange(10):
				data = g_result[0][i]
				data = data.reshape(28, 28)
				data = cmodel.de_normalize(data)
				plt.subplot(2, 5, i)
				plt.imshow(data, cmap='gray')
			plt.show()

		if step >= training_epochs * mnist.train.num_examples / batch_size:
			break

	print 'Training Completed'


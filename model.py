# -*- coding:utf-8 -*-

import math
import random
import numpy
import tensorflow as tf


def relu_layer(layer_input, output_units, name='', dropout=False, keep_prob=None):
	if not isinstance(layer_input, list):
		layer_input = [layer_input]
	input_units = int(layer_input[0].shape[1])
	weights = tf.Variable(
		tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units))),
		# tf.truncated_normal([input_units, output_units], stddev=0.1),
		name=name+'w'
	)
	biases = tf.Variable(tf.zeros([output_units]), name=name+'b')
	layer_output = [tf.nn.relu(tf.matmul(inputs, weights) + biases) for inputs in layer_input]
	if dropout:
		layer_dropout_output = [tf.nn.dropout(outputs, keep_prob) for outputs in layer_output]
		return layer_dropout_output
	else:
		return layer_output


def leaky_relu_layer(layer_input, output_units, name='', dropout=False, keep_prob=None):
	if not isinstance(layer_input, list):
		layer_input = [layer_input]
	input_units = int(layer_input[0].shape[1])
	weights = tf.Variable(
		tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units))),
		# tf.truncated_normal([input_units, output_units], stddev=0.1),
		name=name+'w'
	)
	biases = tf.Variable(tf.zeros([output_units]), name=name+'b')
	layer_output = []
	for inputs in layer_input:
		u = tf.matmul(inputs, weights) + biases
		layer_output.append(tf.maximum(u, 0.1 * u))
	if dropout:
		layer_dropout_output = [tf.nn.dropout(outputs, keep_prob) for outputs in layer_output]
		return layer_dropout_output
	else:
		return layer_output


def sigmoid_layer(layer_input, output_units, name='', dropout=False, keep_prob=None):
	if not isinstance(layer_input, list):
		layer_input = [layer_input]
	input_units = int(layer_input[0].shape[1])
	weights = tf.Variable(
		tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units))),
		# tf.truncated_normal([input_units, output_units], stddev=0.1),
		name=name+'w'
	)
	biases = tf.Variable(tf.zeros([output_units]), name=name+'b')
	layer_output = [tf.nn.sigmoid(tf.matmul(inputs, weights) + biases) for inputs in layer_input]
	if dropout:
		layer_dropout_output = [tf.nn.dropout(outputs, keep_prob) for outputs in layer_output]
		return layer_dropout_output
	else:
		return layer_output


def tanh_layer(layer_input, output_units, name='', dropout=False, keep_prob=None):
	if not isinstance(layer_input, list):
		layer_input = [layer_input]
	input_units = int(layer_input[0].shape[1])
	weights = tf.Variable(
		tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units))),
		# tf.truncated_normal([input_units, output_units], stddev=0.1),
		name=name+'w'
	)
	biases = tf.Variable(tf.zeros([output_units]), name=name+'b')
	layer_output = [tf.nn.tanh(tf.matmul(inputs, weights) + biases) for inputs in layer_input]
	if dropout:
		layer_dropout_output = [tf.nn.dropout(outputs, keep_prob) for outputs in layer_output]
		return layer_dropout_output
	else:
		return layer_output


def softmax_layer(layer_input, output_units, name=''):
	if not isinstance(layer_input, list):
		layer_input = [layer_input]
	input_units = int(layer_input[0].shape[1])
	weights = tf.Variable(
		tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units))),
		# tf.truncated_normal([input_units, output_units], stddev=0.1),
		name=name+'w'
	)
	biases = tf.Variable(tf.zeros([output_units]), name=name+'b')
	layer_output = [tf.nn.softmax(tf.matmul(inputs, weights) + biases) for inputs in layer_input]
	return layer_output


# using leaky_relu to avoid dying relu problem(which makes many zero outputs)
def generative_model(z_inputs, structure, keep_prob, name='g_'):
	hidden_layers = [z_inputs]
	for i, hidden_units in enumerate(structure):
		if i < len(structure) - 1:
			hidden_layers.append(leaky_relu_layer(
				hidden_layers[i], hidden_units, name=name+'layer%d_' % i,
				dropout=keep_prob != 1.0, keep_prob=keep_prob))
		else:
			hidden_layers.append(tanh_layer(
				hidden_layers[i], hidden_units, name=name+'layer%d_' % i))
	return hidden_layers[-1][0]


def discriminative_model(inputs, structure, keep_prob, name='d_'):
	hidden_layers = [inputs]
	for i, hidden_units in enumerate(structure):
		if i < len(structure) - 1:
			hidden_layers.append(leaky_relu_layer(
				hidden_layers[i], hidden_units, name=name+'layer%d_' % i,
				dropout=keep_prob != 1.0, keep_prob=keep_prob))
		else:
			hidden_layers.append(sigmoid_layer(
				hidden_layers[i], hidden_units, name=name+'layer%d_' % i))
	return hidden_layers[-1]


def minimize_generative_model(z_probability, learning_rate, var_list=None):
	# avoid gradients around optimal point being too big(but seems hard to reach optimal)
	# avoid dropping into zero because gradients around zero can be infinite
	# cost = tf.reduce_mean(tf.log(1 - z_probability))
	cost = tf.reduce_mean(-tf.log(z_probability))
	z_accuracy = tf.reduce_mean(tf.cast(tf.greater(z_probability, 0.5), 'float'))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=var_list)
	return cost, z_accuracy, train_step


def maximize_discriminative_model(x_probability, z_probability, learning_rate, var_list=None):
	score = tf.reduce_mean(tf.log(x_probability) + tf.log(1 - z_probability))
	x_accuracy = tf.reduce_mean(tf.cast(tf.greater(x_probability, 0.5), 'float'))
	z_accuracy = tf.reduce_mean(tf.cast(tf.greater(z_probability, 0.5), 'float'))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(-score, var_list=var_list)
	return score, x_accuracy, z_accuracy, train_step


def normalize(data):
	return 2 * data.astype(numpy.float32) - 1


def de_normalize(data):
	return 0.5 * data.astype(numpy.float32) + 0.5


def sample_noise(samples, noise_size):
	# noise_z = [[random.gauss(0, 1) for i in xrange(noise_size)] for j in xrange(samples)]
	# noise_z = [[random.uniform(0, 1) for i in xrange(noise_size)] for j in xrange(samples)]
	noise_z = numpy.random.normal(0, 1, size=(samples, noise_size)).astype(numpy.float32)
	return noise_z

# -*- coding:utf-8 -*-

import math
import random
import numpy
import tensorflow as tf


def relu(inputs):
	if isinstance(inputs, list):
		return [tf.nn.relu(value) for value in inputs]
	else:
		return tf.nn.relu(inputs)


def leaky_relu(inputs):
	if isinstance(inputs, list):
		return [tf.maximum(value, 0.1 * value) for value in inputs]
	else:
		return tf.maximum(inputs, 0.1 * inputs)


def tanh(inputs):
	if isinstance(inputs, list):
		return [tf.nn.tanh(value) for value in inputs]
	else:
		return tf.nn.tanh(inputs)


def linear_layer(layer_input, output_units, name='', dropout=False, keep_prob=None):
	if not isinstance(layer_input, list):
		layer_input = [layer_input]
	input_units = int(layer_input[0].shape[1])
	weights = tf.Variable(
		# tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units))),
		tf.truncated_normal([input_units, output_units], stddev=0.1),
		name=name+'weight'
	)
	biases = tf.Variable(tf.zeros([output_units]), name=name+'bias')
	layer_output = [tf.matmul(inputs, weights) + biases for inputs in layer_input]
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
		# tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units))),
		tf.truncated_normal([input_units, output_units], stddev=0.1),
		name=name+'weight'
	)
	biases = tf.Variable(tf.zeros([output_units]), name=name+'bias')
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
		# tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units))),
		tf.truncated_normal([input_units, output_units], stddev=0.1),
		name=name+'weight'
	)
	biases = tf.Variable(tf.zeros([output_units]), name=name+'bias')
	layer_output = [tf.nn.tanh(tf.matmul(inputs, weights) + biases) for inputs in layer_input]
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
		# tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units))),
		tf.truncated_normal([input_units, output_units], stddev=0.1),
		name=name+'weight'
	)
	biases = tf.Variable(tf.zeros([output_units]), name=name+'bias')
	layer_output = []
	for inputs in layer_input:
		u = tf.matmul(inputs, weights) + biases
		layer_output.append(tf.maximum(u, 0.1 * u))
	if dropout:
		layer_dropout_output = [tf.nn.dropout(outputs, keep_prob) for outputs in layer_output]
		return layer_dropout_output
	else:
		return layer_output


def conv_layer(layer_input, out_channels, filter_height=5, filter_width=5, stride_height=2, stride_width=2, name=''):
	# layer_input: [batch, in_height, in_width, in_channels]
	# filters: [filter_height, filter_width, in_channels, out_channels]
	# return: [batch, out_height, in_height, out_channels]
	if not isinstance(layer_input, list):
		layer_input = [layer_input]
	in_channels = int(layer_input[0].shape[3])
	filters = tf.Variable(
		tf.truncated_normal([filter_height, filter_width, in_channels, out_channels], stddev=0.1),
		name=name+'filter'
	)
	biases = tf.Variable(tf.zeros([out_channels]), name=name+'bias')
	layer_output = [
		tf.nn.conv2d(inputs, filters, strides=[1, stride_height, stride_width, 1], padding='SAME') + biases
		for inputs in layer_input]
	return layer_output


def deconv_layer(
		layer_input, batch_size, out_channels, filter_height=5, filter_width=5, stride_height=2, stride_width=2,
		out_height=0, out_width=0, name=''):
	# layer_input: [batch, in_height, in_width, in_channels]
	# filters: [filter_height, filter_width, out_channels, in_channels]
	# output_shape: [batch, out_height, out_height, out_channels]
	if not isinstance(layer_input, list):
		layer_input = [layer_input]
	in_height = int(layer_input[0].shape[1])
	in_width = int(layer_input[0].shape[2])
	in_channels = int(layer_input[0].shape[3])
	filters = tf.Variable(
		tf.truncated_normal([filter_height, filter_width, out_channels, in_channels], stddev=0.1),
		name=name+'filter'
	)
	biases = tf.Variable(tf.zeros([out_channels]), name=name+'bias')
	out_height = out_height or in_height * stride_height
	out_width = out_width or in_width * stride_width
	output_shape = [batch_size, out_height, out_width, out_channels]
	layer_output = [
		tf.nn.conv2d_transpose(
			inputs, filters, output_shape=output_shape, strides=[1, stride_height, stride_width, 1], padding='SAME')
		+ biases for inputs in layer_input]
	return layer_output


def generative_model(z_inputs, batch_size, structure, name='g_'):
	# input: [batch, noise_size] --> [[batch, height, width, channels]...]
	# structure: [[channels, height, width],
	# [out_channels, filter_height, filter_width, stride_width, stride_height, out_height, out_width]...]
	hidden_layers = [z_inputs]
	for i, args in enumerate(structure):
		if i == 0:
			output_units = args[0] * args[1] * args[2]
			hidden_layers.append(linear_layer(hidden_layers[i], output_units, name=name + 'layer%d_' % i))
			hidden_layers[-1] = [tf.reshape(layer, [-1, args[1], args[2], args[0]]) for layer in hidden_layers[-1]]
		else:
			hidden_layers.append(deconv_layer(hidden_layers[i], batch_size, *args, name=name + 'layer%d_' % i))
			if i < len(structure) - 1:
				hidden_layers[-1] = relu(hidden_layers[-1])
			else:
				hidden_layers[-1] = tanh(hidden_layers[-1])
	return hidden_layers[-1][0]


def discriminative_model(inputs, structure, keep_prob, name='d_'):
	# input: [[batch, in_height, in_width, in_channels]...]
	# structure: [[out_channels, filter_height, filter_width, stride_height, stride_width]...[1]]
	hidden_layers = [inputs]
	for i, args in enumerate(structure):
		if len(args) > 2:
			# conv_layer
			hidden_layers.append(conv_layer(hidden_layers[i], *args, name=name+'layer%d_' % i))
			hidden_layers[-1] = leaky_relu(hidden_layers[-1])
		elif len(args) == 2:
			# 1st full-connected layer
			layer_height = int(hidden_layers[i][0].shape[1])
			layer_width = int(hidden_layers[i][0].shape[2])
			layer_channels = int(hidden_layers[i][0].shape[3])
			hidden_layers[i] = [
				tf.reshape(layer, [-1, layer_height * layer_width * layer_channels]) for layer in hidden_layers[i]]
			if i < len(structure) - 1:
				hidden_layers.append(leaky_relu_layer(
					hidden_layers[i], args[1], name=name+'layer%d_' % i,
					dropout=keep_prob != 1.0, keep_prob=keep_prob))
			else:
				hidden_layers.append(sigmoid_layer(
					hidden_layers[i], args[1], name=name + 'layer%d_' % i))
		elif i < len(structure) - 1:
			hidden_layers.append(leaky_relu_layer(
				hidden_layers[i], args[0], name=name + 'layer%d_' % i,
				dropout=keep_prob != 1.0, keep_prob=keep_prob))
		else:
			hidden_layers.append(sigmoid_layer(
				hidden_layers[i], args[0], name=name + 'layer%d_' % i))
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
	noise_z = numpy.random.uniform(-1, 1, size=(samples, noise_size)).astype(numpy.float32)
	return noise_z


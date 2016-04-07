#Deep reader: two layer LSTM concatenated into g
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.core import *
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.utils.visualize_util import plot
import theano

import sys
import numpy as np

def get_y_T(X):
	return X[:, -1, :]

def get_query(X):
	return X[:, 2000:, :]

def get_doc(X):
	return X[:, :2000, :]

def get_R(X):
	Y, s_att = X.values()
	exchange = K.permute_dimensions(Y, (0,) + (2,1))
	ans = K.T.batched_dot(exchange, s_att)
	return ans

# maxlen = 2500;
# maxdoclen = 2000;
# maxquerylen = 500;
# vocab_size = 25000;

def train(x_train, y_train, x_test, y_test, vocab_size, maxlen, maxdoclen, maxquerylen):
	model = Graph()
	model.add_input(name = 'input', input_shape = (maxlen,), dtype = 'int')
	model.add_node(Embedding(vocab_size,256, input_length=maxlen, dropout=0.5), input='input', name='emb')   #single embedding layer, then separate query and doc?
	model.add_node(Lambda(get_doc, output_shape = (maxdoclen, 256)), input = 'emb', name = 'doc')
	model.add_node(Lambda(get_query, output_shape = (maxquerylen, 256)), input = 'emb', name = 'query')
	model.add_node(LSTM(128, dropout_W=0.1, dropout_U=0.1, return_sequences = True), input = 'doc', name = 'lstm1')
	model.add_node(LSTM(128, dropout_W=0.1, dropout_U=0.1, return_sequences = True, go_backwards = True), input = 'doc', name = 'lstm2')
	model.add_node(Dropout(0.1), inputs = ['lstm1', 'lstm2'], name = 'dropout2')
	model.add_node(LSTM(128, dropout_W=0.1, dropout_U=0.1), input = 'query', name = 'lstm3')
	model.add_node(LSTM(128, dropout_W=0.1, dropout_U=0.1, go_backwards = True), input = 'query', name = 'lstm4')
	model.add_node(Dropout(0.1), inputs = ['lstm3', 'lstm4'], name = 'dropout4')
	# model.add_node(Lambda(get_y_T, output_shape = (128,)), input = 'dropout1', name = 'slice')
	model.add_node(TimeDistributedDense(128), input = 'dropout2', name = 'tdd')
	model.add_node(Dense(128), input = 'dropout4', name = 'querydense')
	model.add_node(RepeatVector(maxdoclen), input = 'querydense', name = 'querymatrix')
	model.add_node(Activation('tanh'), inputs = ['tdd', 'querymatrix'], name = 'm_t')
	model.add_node(TimeDistributedDense(1), input = 'm_t', name = 'pre_s')
	model.add_node(Flatten(), input = 'pre_s', name = 'flattened_pre_s')
	model.add_node(Activation('softmax'), input = 'flattened_pre_s', name = 's_att')
	model.add_node(Reshape((maxdoclen,1)), input = 's_att', name = 's_att_res')
	model.add_node(Lambda(get_R, output_shape = (maxdoclen,1)), inputs = ['dropout2', 's_att'], merge_mode = 'join', name = 'weighted_r_res')
	model.add_node(Reshape((maxdoclen,)), input = 'weighted_r_res', name = 'weighted_r')
	model.add_node(Dense(128), input = 'dropout4', name = 'W_g_u')
	model.add_node(Dense(128), input = 'weighted_r', name = 'W_g_r')
	model.add_node(Activation('tanh'), inputs = ['W_g_u', 'W_g_r'], merge_mode = 'sum', name = 'embed_g')
	model.add_node(Dense(vocab_size, activation = 'softmax'), input = 'embed_g', name = 'output', create_output = True)
	print model.summary()
	model.compile(loss = {'output': 'categorical_crossentropy'}, optimizer = 'rmsprop')
	plot(model, to_file='model.png', show_shape = True)
	print('Train...')
	# plot(model, to_file='model.png', show_shape = True)
	model.fit({'input': x_train, 'output': y_train}, batch_size=batch_size, nb_epoch=15, validation_data={'input': x_test, 'output': y_test}, show_accuracy=True)
	score, acc = model.evaluate({'input': x_test, 'output': y_test},
	                            batch_size=batch_size,
	                            show_accuracy=True)
	print('Test score:', score)
	print('Test accuracy:', acc)



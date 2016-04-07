#Deep reader: two layer LSTM concatenated into g
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Lambda, Merge, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.utils.visualize_util import plot
import theano

import sys
import numpy as np

def get_y_T(X):
	return X[:, -1, :]

def train(x_train, y_train, x_test, y_test, vocab_size):
	model = Graph()
	model.add_input(name = 'input', input_shape = (maxlen,), dtype = 'int')
	model.add_node(Embedding(vocab_size,256, input_length=maxlen, dropout=0.5), input='input', name='emb')
	model.add_node(LSTM(128, dropout_W=0.1, dropout_U=0.1, return_sequences = True), input = 'emb', name = 'lstm1')
	model.add_node(Dropout(0.1), input = 'lstm1', name = 'dropout1')
	model.add_node(LSTM(128, dropout_W=0.1, dropout_U=0.1), input = 'dropout1', name = 'lstm2')
	model.add_node(Dropout(0.1), input = 'lstm2', name = 'dropout2')
	model.add_node(Lambda(get_y_T, output_shape = (128,)), input = 'dropout1', name = 'slice')
	model.add_node(Dense(vocab_size), inputs = ['slice', 'dropout2'], name = 'dense', merge_mode = 'concat', concat_axis = 1)
	model.add_node(Activation('softmax'), input = 'dense', name = 'softmax')
	model.add_output(name='output', input='softmax')
	print model.summary()
	model.compile(loss = {'output': 'categorical_crossentropy'}, optimizer = 'rmsprop')
	plot(model, to_file='model2.png', show_shape = True)
	print('Train...')
	plot(model, to_file='model.png', show_shape = True)
	model.fit({'input': x_train, 'output': y_train}, batch_size=batch_size, nb_epoch=15, validation_data={'input': x_test, 'output': y_test}, show_accuracy=True)
	score, acc = model.evaluate({'input': x_test, 'output': y_test},
	                            batch_size=batch_size,
	                            show_accuracy=True)
	print('Test score:', score)
	print('Test accuracy:', acc)

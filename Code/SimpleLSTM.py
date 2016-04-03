#single LSTM layer, qca
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

def train(x_train, y_train, x_test, y_test):
	model = Sequential()
	model.add(Embedding(vocab_size, 128, input_length=maxlen, dropout=0.5, mask_zero = True))
	model.add(LSTM(128, dropout_W=0.5, dropout_U=0.1))  # try using a GRU instead, for fun
	model.add(Dropout(0.5))
	model.add(Dense(vocab_size))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print('Train...')
	batch_size = 32
	plot(model, to_file='model.png', show_shape = True)
	model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=15,
	          validation_data=(x_test, y_test), show_accuracy=True)
	score, acc = model.evaluate(x_test, y_test,
	                            batch_size=batch_size,
	                            show_accuracy=True)
	print('Test score:', score)
	print('Test accuracy:', acc)



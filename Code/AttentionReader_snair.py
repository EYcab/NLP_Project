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
	model = Graph()
	model.add_input(name = 'input', input_shape = (maxlen,), dtype = 'int')
	model.add_node(Embedding(vocab_size,256, input_length=maxlen, dropout=0.5), input='input', name='emb')

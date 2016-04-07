from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Lambda, Merge, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
# from keras.utils.visualize_util import plot
import theano

import sys
import numpy as np

if __name__ == '__main__':
	f = open('../data/validation_set.txt')   #modify according to what you want
	raw_data = f.readlines()
	input_doc = []                        # list of list of words in doc
	input_query = []                      # list of list of words in query
	target_word = []                      # list of target words
	vocab = {}                            # vocabulary
	url = []               				  # list of urls, mostly useless
	entity_list = []                      # entity listing for each doc, mostly useless
	vocab_limit = 100000                   # Will depend on training data
	i = 0

	while i < len(raw_data):
		url.append(raw_data[i].strip('\n'))
		i += 2
		input_doc.append(raw_data[i].strip('\n').split(' '))
		i += 2
		input_query.append(raw_data[i].strip('\n').split(' '))
		i += 2
		target_word.append(raw_data[i].strip('\n'))
		i += 2
		entity_list.append([])
		while(raw_data[i] != '\n'):
			entity_list[-1].append(raw_data[i].strip('\n'))
			i += 1
		i += 2
	# l = {}
	for doc in input_doc + input_query:
		for word in doc:
			# if(word.startswith('@entity')):
			# 	l[word[7:]] = 1
			if word not in vocab:
				vocab[word] = 1
			else:
				vocab[word] += 1
	# print sorted([int(m) for m in l])
	vocablist = vocab.items()
	vocablist.sort(key = lambda t: -t[1])
	vocab = {}
	vocab_size = 0
	#TODO: can collaspe some prepositions into one
	
	vocab[2] = '@placeholder'
	for j in range(1000):
		vocab['@entity' + str(j)] = j + 3
	vocab_size = 1000
	for word in vocablist:
		if word[0] in vocab:
			continue
		if(vocab_size >= vocab_limit):
			vocab[word[0]] = 1  #reserve 1 for OOV, reserve 0 for padding
		else:
			vocab[word[0]] = vocab_size + 3
			vocab_size += 1
	for i, doc in enumerate(input_doc):
		for j, word in enumerate(doc):
			if word in vocab:
				input_doc[i][j] = vocab[word]
			else:
				input_doc[i][j] = 1

	for i, query in enumerate(input_query):
		for j, word in enumerate(query):
			if word in vocab:
				input_query[i][j] = vocab[word]
			else:
				input_query[i][j] = 1
	target_word = [vocab[word] for word in target_word]  #assuming every entity is in vocab
	#for Deep reader
	inputs = [query + [vocab_size + 4] + doc for doc, query in zip(input_doc, input_query)]
	vocab_size = vocab_size + 5
	maxlen = 2000
	frac = 0.8
	x_train = sequence.pad_sequences(inputs, maxlen=maxlen)
	x_test = x_train[int(frac*len(target_word)):]
	x_train = x_train[:int(frac*len(target_word))]
	y_train = np.zeros((len(target_word), vocab_size))
	y_train[np.arange(len(y_train)), np.array(target_word)] = 1
	y_test = y_train[int(frac*len(target_word)):]
	y_train = y_train[:int(frac*len(target_word))]
	batch_size = 32

	# model = Sequential()
	# model.add(Embedding(vocab_size, 128, input_length=maxlen, dropout=0.5, mask_zero = True))
	# model.add(LSTM(128, dropout_W=0.5, dropout_U=0.1))  # try using a GRU instead, for fun
	# model.add(Dropout(0.5))
	# model.add(Dense(vocab_size))
	# model.add(Activation('softmax'))
	# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	# print('Train...')
	# batch_size = 32
	# plot(model, to_file='model.png', show_shape = True)
	# model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=15,
	#           validation_data=(x_test, y_test), show_accuracy=True)
	# score, acc = model.evaluate(x_test, y_test,
	#                             batch_size=batch_size,
	#                             show_accuracy=True)
	# print('Test score:', score)
	# print('Test accuracy:', acc)

	
	def get_y_T(X):
		return X[:, -1, :]

	model = Graph()
	model.add_input(name = 'input', input_shape = (maxlen,), dtype = 'int')
	model.add_node(Embedding(vocab_size,256, input_length=maxlen, dropout=0.1), input='input', name='emb')
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

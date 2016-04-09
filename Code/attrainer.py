from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Lambda, Merge, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.utils.visualize_util import plot
import theano

import sys
import numpy as np

if __name__ == '__main__':
		
	def process_data(file, training_data = True, vocab = None, vocab_size = None):
		f = open(file)                        
		raw_data = f.readlines()
		input_doc = []                        # list of list of words in doc
		input_query = []                      # list of list of words in query
		target_word = []                      # list of target words
		vocab = {}                            # vocabulary
		# url = []               				  # list of urls, mostly useless
		# entity_list = []                      # entity listing for each doc, mostly useless
		vocab_limit = 50000                   # Will depend on training data
		i = 0

		while i < len(raw_data):
			url.append(raw_data[i].strip('\n'))
			i += 2
			input_doc.append(raw_data[i].strip('\n').split(' '))
			i += 2
			input_query.append(raw_data[i].strip('\n').split(' '))
			i += 2
			# target_word.append(raw_data[i].strip('\n'))
			i += 2
			entity_list.append([])
			while(raw_data[i] != '\n'):
				# entity_list[-1].append(raw_data[i].strip('\n'))
				i += 1
			i += 2
		# l = {}
		if not vocab:
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
		
			vocab['@placeholder'] = 2
			for j in range(1000):
				vocab['@entity' + str(j)] = j + 3
			vocab_size = 1003
			for word in vocablist:
				if word[0] in vocab:
					continue
				if(vocab_size >= vocab_limit):
					vocab[word[0]] = 1  #reserve 1 for OOV, reserve 0 for padding
				else:
					vocab[word[0]] = vocab_size
					vocab_size += 1
			vocab['#delim'] = vocab_size
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
		# inputs = [query + [vocab['#delim']] + doc for doc, query in zip(input_doc, input_query)]
		return input_doc, input_query, target_word, vocab, vocab_size
	
	train_file = '/home/ee/btech/ee1130504/data/cnn_training_set.txt'
	valid_file = '/home/ee/btech/ee1130504/data/cnn_validation_set.txt'
	test_file = '/home/ee/btech/ee1130504/data/cnn_test_set.txt'

	train_input_doc, train_input_query, train_target_word, vocab, vocab_size = process_data(train_file)
	valid_input_doc, valid_input_query, valid_target_word, vocab, vocab_size = process_data(valid_file, False, vocab, vocab_size)
	test_input_doc, test_input_query, test_target_word, vocab, vocab_size = process_data(test_file, False, vocab, vocab_size)

	maxdoclen = max([len(doc) for doc in train_input_doc])
	maxquerylen = max([len(query) for query in train_input_query])
	maxlen = maxdoclen + maxquerylen
	# frac = 0.8
	# train_inputs = [query + [vocab['#delim']] + doc for doc, query in zip(train_input_doc, train_input_query)]
	# valid_inputs = [query + [vocab['#delim']] + doc for doc, query in zip(valid_input_doc, valid_input_query)]
	# test_inputs = [query + [vocab['#delim']] + doc for doc, query in zip(test_input_doc, test_input_query)]

	batch_size = 32
	
	def generate_training_batches():
		index = 0         #IF I understand correctly, state of index will be saved because it's local
		while True:
			remaining = len(train_input_doc) - index
			input_slice = []
			target_slice = []
			if remaining >= batch_size:
				input_doc_slice = train_input_doc[index:(index + batch_size)]
				input_query_slice = train_input_query[index:(index + batch_size)]
				target_slice = train_target_word[index:(index + batch_size)]
				index += batch_size
			else:
				input_doc_slice = train_input_doc[index:]
				input_doc_slice += train_input_doc[:(batch_size - remaining)]
				input_query_slice = train_input_query[index:]
				input_query_slice += train_input_query[:(batch_size - remaining)]
				target_slice = train_target_word[index:]
				target_slice += train_target_word[:(batch_size - remaining)]
				index = batch_size - remaining
			x_train_doc = sequence.pad_sequences(input_doc_slice, maxlen = maxdoclen)
			x_train_query = sequence.pad_sequences(input_query_slice, maxlen = maxquerylen)
			x_train = np.concatenate((x_train_doc, x_train_query), axis = 1)
			y_train = np.zeros((batch_size, vocab_size))
			y_train[np.arange(batch_size), np.array(target_slice)] = 1
			yield {'input': x_train, 'output:' y_train}

	def generate_valid_batches():
		index = 0
		while True:
			remaining = len(valid_input_doc) - index
			input_slice = []
			target_slice = []
			if remaining >= batch_size:
				input_doc_slice = valid_input_doc[index:(index + batch_size)]
				input_query_slice = valid_input_query[index:(index + batch_size)]
				target_slice = valid_target_word[index:(index + batch_size)]
				index += batch_size
			else:
				input_doc_slice = valid_input_doc[index:]
				input_doc_slice += valid_input_doc[:(batch_size - remaining)]
				input_query_slice = valid_input_query[index:]
				input_query_slice += valid_input_query[:(batch_size - remaining)]
				target_slice = valid_target_word[index:]
				target_slice += valid_target_word[:(batch_size - remaining)]
				index = batch_size - remaining
			x_valid_doc = sequence.pad_sequences(input_doc_slice, maxlen = maxdoclen)
			x_valid_query = sequence.pad_sequences(input_query_slice, maxlen = maxquerylen)
			x_valid = np.concatenate((x_valid_doc, x_valid_query), axis = 1)
			y_valid = np.zeros((batch_size, vocab_size))
			y_valid[np.arange(batch_size), np.array(target_slice)] = 1
			yield {'input': x_valid, 'output:' y_valid}

	def generate_test_batches():
		index = 0
		while True:
			remaining = len(test_input_doc) - index
			input_slice = []
			target_slice = []
			if remaining >= batch_size:
				input_doc_slice = test_input_doc[index:(index + batch_size)]
				input_query_slice = test_input_query[index:(index + batch_size)]
				target_slice = test_target_word[index:(index + batch_size)]
				index += batch_size
			else:
				input_doc_slice = test_input_doc[index:]
				input_doc_slice += test_input_doc[:(batch_size - remaining)]
				input_query_slice = test_input_query[index:]
				input_query_slice += test_input_query[:(batch_size - remaining)]
				target_slice = test_target_word[index:]
				target_slice += test_target_word[:(batch_size - remaining)]
				index = batch_size - remaining
			x_test_doc = sequence.pad_sequences(input_doc_slice, maxlen = maxdoclen)
			x_test_query = sequence.pad_sequences(input_query_slice, maxlen = maxquerylen)
			x_test = np.concatenate((x_test_doc, x_test_query), axis = 1)
			y_test = np.zeros((batch_size, vocab_size))
			y_test[np.arange(batch_size), np.array(target_slice)] = 1
			yield {'input': x_test, 'output:' y_test}

	def get_y_T(X):
		return X[:, -1, :]

	def get_query(X):
		return X[:, maxquerylen:, :]

	def get_doc(X):
		return X[:, :maxdoclen, :]

	def get_R(X):
		Y, s_att = X.values()
		exchange = K.permute_dimensions(Y, (0,) + (2,1))
		ans = K.T.batched_dot(exchange, s_att)
		return ans

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
	# plot(model, to_file='model2.png', show_shape = True)
	print('Train...')
	# plot(model, to_file='model.png', show_shape = True)
	callbacks = []
	model_path = '/home/ee/btech/ee1130504/Models/AttentionReader/'
	callbacks.append(ModelCheckpoint(model_path + 'model_weights.{epoch:02d}-{val_loss:.3f}.hdf5', verbose = 1))
	callbacks.append(EarlyStopping(patience = 5))
	model.fit_generator(generate_training_batches, samples_per_epoch = 200, nb_epoch = 5000, validation_data = generate_valid_batches, nb_valid_samples = len(valid_input_doc) / batch_size, callbacks = callbacks, show_accuracy = True)
	score, acc = model.evaluate_generator(generate_test_batches, nb_valid_samples = len(test_input_doc) / batch_size, show_accuracy = True)
	print('Test score:', score)
	print('Test accuracy:', acc)

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

import sys
import numpy as np

if __name__ == '__main__':
	f = open('data/validation_set.txt')   #modify according to what you want
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
	vocab_size += 1
	inputs = [query + [vocab_size + 4] + doc for doc, query in zip(input_doc, input_query)]
	maxlen = 2000
	frac = 0.8
	x_train = sequence.pad_sequences(inputs, maxlen=maxlen)
	x_test = x_train[int(frac*len(x_train)):]
	x_train = x_train[:int(frac*len(x_train))]
	y_train = np.zeros((len(target_word), vocab_size))
	y_train[np.arange(len(y_train)), np.array(target_word)] = 1
	y_test = y_train[int(frac*len(x_train)):]
	y_train = y_train[:int(frac*len(x_train))]
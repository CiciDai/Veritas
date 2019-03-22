import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.preprocessing import LabelEncoder

import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk import pos_tag, ne_chunk
#nltk.download('words')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')

import os
import pickle
import socket
from multiprocessing import Queue, Value


def text_to_tokens(words, word_index):
	output = []
	for word in words:
		index = word_index.get(word, 1)
		output.append(int(index))
	return [output]

# lemmatize words
def lemmatizer(words):
	lemmatizer=WordNetLemmatizer()
	output =[lemmatizer.lemmatize(word, pos='v') for word in words]
	return output

# stemming words
def stemmer(words):
	porter_stemmer = PorterStemmer()
	result = []
	for word in words:
		if str.lower(word) == word:
			word = porter_stemmer.stem(word)
		result.append(word)
	return result

# remove stopwords
def remove_stop_words(words):
	stop_words = set(stopwords.words('english'))
	stop_words.remove('no')
	stop_words.remove('not')
	output =[word for word in words if not word in stop_words]
	return output

# named-entity recognition
def recognizer(words):
	tagged = pos_tag(words)
	grammar = ('''
		NP: {<DT>?<JJ>*<NN>}
		''')
	chunkParser = nltk.RegexpParser(grammar)
	tree = chunkParser.parse(tagged)
	return tree
	
def check_sentiment(sentence, text_len, lstm_model, word_index, data_classes):
	tokens = text_to_word_sequence(sentence)
	output = remove_stop_words(tokens)
#	output = str(recognizer(tokens))
#	output = text_to_word_sequence(output, lower=False)
	output = lemmatizer(output)
	output = stemmer(output)
#	output = tokenizer.texts_to_sequences([output])
	output = text_to_tokens(output, word_index)
	output = tf.keras.preprocessing.sequence.pad_sequences(output,
														padding = 'post',
														value=word_index["<PAD>"],
														maxlen=text_len)

	probs = lstm_model.predict(output)
	guess = np.argmax(probs, axis=-1)
	result = data_classes[guess[0]]
	return result, probs[0][guess[0]]

	
def startNLP(NLPQueue, end):
	text_len = 32
	data_classes = ['anger','disgust','fear','guilt','joy','sadness','shame' ]

	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	word_index = tokenizer.word_index
	
	lstm_model = tf.keras.models.load_model("NLPmodel_100d_32.h5")
	
	lstm_model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])

	word_index = {k:(v+1) for k,v in word_index.items()} 
	word_index['<PAD>']=0
	word_index["<UNK>"] = 1
	#print(word_index)

	recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	recv_socket.bind(('0.0.0.0', 8002))
	recv_socket.listen(0)
	recv_conn = recv_socket.accept()[0]
	
	print("speech recv conn established")
	
	while end.value == 0:
		recv_msg = recv_conn.recv(512).decode()
		#sentence = 'I feel I will fail the circuit exam on Friday'
		timeSpeech = recv_msg.split(':')
		if len(timeSpeech) > 1:
			print("got message: " + timeSpeech[1])
			result, prob = check_sentiment(timeSpeech[1], text_len, lstm_model, word_index, data_classes)
			result += ":" + str(prob)
			print(result)
			NLPQueue.put((timeSpeech[0], result, timeSpeech[1]))

	print("ending NLP")
	

#if __name__ == '__main__':
#	end = Value('i', 0)
#	startNLP(end)
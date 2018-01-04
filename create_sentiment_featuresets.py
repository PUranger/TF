#[chair, table, spoon, television]
#I pulled the chair up to the table
#np.zeros(len(lexicon))   #[0 0 0 0]

import nltk
from nltk.tokenize import word_tokenize
#function e.g. for nltk.tokenize:
#Before: I pulled the chair up to the table
#after: [I, pulled, the, chair, up, to, the, table]
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos,neg):
	lexicon = []
	# fi for "file"
	for fi in [pos,neg]:
		with open(fi,'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())  #lowercased
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)   #w_counts is a dictionary
	#w_counts = {'the':52521, 'and':25242}

	l2 = []
	for w in w_counts:
		#might need to change 1000 and 50. This is used for
		#removing words that are too frequent (like or, and, etc)
		#or too less, which we don't care about.

		#Since w_counts is a dictionary, this if condition compare the contents in each
		#class. e.g.  for 'the', decide whether 1000>52521>50.
		if 1000 > w_counts[w] > 50:
			l2.append(w)   #append the word in l2
	print(len(l2))
	#print(l2)	
	return l2   #l2 is a list with all words that we care about, string list
	#After the above function, we have a word list containing all key words from two files. 

def sample_handling(sample, lexicon, classification):
	featureset = []

	with open(sample,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:   #for each line:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:  #for each word in one line:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			featureset.append([features, classification])

	return featureset

def create_feature_sets_and_labels(pos,neg,test_size=0.1):
	lexicon = create_lexicon(pos,neg)
	features = []
	features += sample_handling('pos.txt',lexicon,[1,0])
	features += sample_handling('neg.txt',lexicon,[0,1])
	#use random.shuffle in order to have the following argmax function true
	#does tf.argmax([output]) == tf.argmax([expectations])
	#and also for statistic reasons.
	random.shuffle(features)

	features = np.array(features)

	testing_size = int(test_size*len(features))

	#first 90%
	#So train_x here is: [[a],[b],[c],...,[n]], a,b,c and n all with len(lexicon)
	#Also, a,b,c and n are all float lists recording how many times words exist
	#in the lexicon list.
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	#last 10%
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y

if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)




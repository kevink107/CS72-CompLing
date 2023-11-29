#==================================================================================
# Dartmouth College, LING48, Spring 2023
# Rolando Coto-Solano (Rolando.A.Coto.Solano@dartmouth.edu)
# Examples for Homework 4.3: Naïve Bayes Classification
#==================================================================================

import itertools
import collections
from nltk import word_tokenize
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.metrics.scores import precision, recall, f_measure
from nltk.collocations import BigramCollocationFinder


# Function to construct a bag of words with both unigrams and bigrams
# https://streamhacker.com/2010/05/24/
# text-classification-sentiment-analysis-stopwords-collocations/
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
  
  bigram_finder = BigramCollocationFinder.from_words(words)
  bigrams = bigram_finder.nbest(score_fn, n)

  tupledWords = []
  for w in words:
    tempList = []
    tempList.append(w)
    tempTuple = tuple(tempList)
    tupledWords.append(tempTuple)

  return dict([(ngram, True) for ngram in itertools.chain(tupledWords, bigrams)])

	
# We will store the negative and positive reviews here	
posReviewsText = []
negReviewsText = []


# Open the file containing the reviews
file = open("mini-movie-reviews.txt", "r") 
fileLines = file.readlines()	


# Go through the file and find the positive and
# negative reviews. Put the text of the reviews
# in the correct list.
for l in fileLines:
	tempLine = l.split("\t")
	if (tempLine[0] == "neg"):
		negReviewsText.append(tempLine[2])
	elif (tempLine[0] == "pos"):
		posReviewsText.append(tempLine[2])

		

		
# This will contain the bag-of-words
# for positive and negative reviews.
negfeats = []
posfeats = []


# for every positive review:
# (1) tokenize it, (2) extract the bag-of-words as
# features, and (3) append it to the positive features.
for f in posReviewsText:
	tokens = word_tokenize(f)
	wordFeats = bigram_word_feats(tokens)
	posfeats.append((wordFeats, 'pos'))
	

# for every negative review:
# (1) tokenize it, (2) extract the bag-of-words as
# features, and (3) append it to the negative features.
for f in negReviewsText:
	tokens = word_tokenize(f)
	wordFeats = bigram_word_feats(tokens)
	negfeats.append((wordFeats, 'neg'))
	
# Get the number of elements that
# will be in the training set.
negcutoff = int(len(negfeats)*0.9) # The number has to be an entire integer so that we can use it as an index
poscutoff = int(len(posfeats)*0.9)

# Make the training and testing sets.
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print('train on ' + str(len(trainfeats)) + ' instances, test on ' + str(len(testfeats)) + ' instances')

# Make a classifier based on the training features.
classifier = NaiveBayesClassifier.train(trainfeats)


# create two blank dictionaries that will contain
# the goldLabels and the predictedLabels
goldLabels = collections.defaultdict(set)
predictedLabels = collections.defaultdict(set)


# get the gold labels and the model predictions
# for every item in the test set and put the
# labels and the predictions in a Python dictionary
for i, (feats, label) in enumerate(testfeats):
		# add the gold labels to the goldLabels dictionary
		goldLabels[label].add(i)
		# get the model's predictions (the "observed" labels)
		observed = classifier.classify(feats)
		# add the model predictions to the predictedLabels dictionary
		predictedLabels[observed].add(i)
		
# Calculate the precision ,recall and
# F for the positive and negative sets.

posPrecision = precision(goldLabels['pos'], predictedLabels['pos'])
posRecall    = recall(goldLabels['pos'], predictedLabels['pos'])
negPrecision = precision(goldLabels['neg'], predictedLabels['neg'])
negRecall    = recall(goldLabels['neg'], predictedLabels['neg'])
negF         = f_measure(goldLabels['neg'], predictedLabels['neg'])
posF         = f_measure(goldLabels['pos'], predictedLabels['pos'])
			
			
# Print the accuracy, precisions, recalls and F values.
print('accuracy:      ' + str(nltk.classify.util.accuracy(classifier, testfeats)))
print('pos precision: ' + str(posPrecision))
print('pos recall:    ' + str(posRecall))
print('neg precision: ' + str(negPrecision))
print('neg recall:    ' + str(negRecall) )
print('neg F-measure: ' + str(negF))
print('pos F-measure: ' + str(posF))
	

# Print the most informative features.	
classifier.show_most_informative_features(n=50)
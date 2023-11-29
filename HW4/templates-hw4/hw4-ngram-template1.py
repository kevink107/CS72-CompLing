#=======================================================================
# Dartmouth College, LING48, Spring 2023
# Rolando Coto-Solano (Rolando.A.Coto.Solano@dartmouth.edu)
# Examples for Exercise 4.1: N-gram probabilities and n-gram text generation
#
# You must study the links below and attempt to modify
# the program according to the homework instructions.
#
# Documentation of the NLTK.LM package
# https://www.nltk.org/api/nltk.lm.html
#
# How to extract n-gram probabilities 
# https://stackoverflow.com/questions/54962539/how-to-get-the-probability-of-bigrams-in-a-text-of-sentences
#=======================================================================

import os
import requests
import io 
import random
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE, NgramCounter, Vocabulary
from nltk.util import ngrams
from collections import Counter
from nltk import word_tokenize, sent_tokenize, bigrams, trigrams

# Open file
file = io.open('english-sherlock.txt', encoding='utf8')
text = file.read()
			  
# Preprocess the tokenized text for language modelling
# https://stackoverflow.com/questions/54959340/nltk-language-modeling-confusion
n = 3
paddedLine = [list(pad_both_ends(word_tokenize(text.lower()), n))]
train, vocab = padded_everygram_pipeline(n, paddedLine)

# Train an n-gram maximum likelihood estimation model.
model = MLE(n) 
model.fit(train, vocab)
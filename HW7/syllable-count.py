#=========================================================================
# Dartmouth College, LING48, Spring 2023
# Rolando Coto-Solano (Rolando.A.Coto.Solano@dartmouth.edu)
# Examples for Homework 7.3: Counting syllables of a word in English
#
# This short example uses a package called PyPhen. It uses hunspell
# dictionaries to split the syllables of a word. If you want to use
# it, you'll need to install this package:
#
# From Anaconda:    conda install -c conda-forge pyphen
# From Colab:       !pip install pyphen
#=========================================================================

import pyphen
dic = pyphen.Pyphen(lang='en')

wordToSplit = "sunrises"

sylls = dic.inserted(wordToSplit)
nsylls = sylls.count("-") + 1


print(wordToSplit)
print(nsylls)
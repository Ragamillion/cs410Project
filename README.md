# cs410Project
Final project for cs410
Tom Phelan and Akhila Prodduturi

This is a context based topic analysis tool for use with collections of tweets, written in Python. 
The tool has been developed using the cc-Mix model proposed in the paper "A cross-collection 
mixture model for comparative text mining" by ChengXiang Zhai, Atulya Velivelli, and Bei Yu, 2004. 
The goalis to allow the user to provide a csv file with the documents in one column and the contextual
variables in a series of other columns in the dataset. The tool then allows the user to clean up
the documents and analyse them. It produces a user-specified number of topic models.

Requirements: 
This tool requires following packages to be installed in Python.
numpy
pandas
nltk
sklearn.feature_extraction.text
TextBlob 
nltk.tokenize
nltk.corpus
autocorrect
html2text
scipy.special
gc

Instructions:

To instantiate the ccMix object, call ccMix(filepath, documentCol, contextDiv, stopwords, nTopics):
filepath is the path to the csv file that contains the documents and contexts
documentCol is the column header of the column containing the documents
contextDiv is an array containing the column headers of the columns with contexts
stopwords is the filepath of the stopwords file to exclude from analysis (standard file included in
this repository)
nTopics is an integer representing the number of topics to create

The methods of the ccMix object are as follows:
buildContext() - required and must be run first
prepareDocs() - optional 
sentimentAnalysis()
buildCorpus()
getTopics()
printTopics(n)
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:59:26 2018
@author: Tom
@author: Akhila
"""
import numpy
import pandas
import nltk
import collections


nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from autocorrect import spell
from html2text import unescape

class InputFile(object):
    # class for the input file and related processing
    filepath = ""
    contextList = ""

    def __init__(self, filepath, documentCol, contextDiv, stopwords):
        # initialize the object. Here filepath is the path to the data file, documentCol is the column in which the documents
        # are found, contextDiv is an array of the columns that determine the contexts, and stopwords is a .txt file containing
        # the stopwords
        self.filepath = filepath
        self.documentCol = documentCol
        self.contextDiv = contextDiv
        self.docfile = pandas.read_csv(self.filepath)
        self.stopwords = stopwords

    def buildContext(self):
        # builds the list of contexts, and creates a new column in the dataframe that contains the combined context for each
        # row. If there are any combinations of contexts that do not exist in the data, they will not be found in the list.

        filepath = "filteredtweetscombined.csv"  # for testing purposes
        # docfile = pandas.read_csv(filepath)

        self.docfile["plsaContext"] = ""
        contextDiv = pandas.Series(["account_category", "dateSent"])

        for context in contextDiv:
            self.docfile["plsaContext"] = self.docfile["plsaContext"] + "|" + self.docfile[context]

        self.docfile["plsaContext"] = self.docfile["plsaContext"].str[1:]
        self.contextList = self.docfile["plsaContext"].unique()

    def prepareDocs(self):
        # escaping html characters
        self.docfile[self.documentCol] = self.docfile[self.documentCol].apply(unescape)

        # unicode to ascii

        # tokenise tweet
        tokenizer = TweetTokenizer()
        self.docfile[self.documentCol] = self.docfile[self.documentCol].str.replace(';', ' ').str.replace('\'', ' ')
        self.docfile[self.documentCol] = self.docfile[self.documentCol].str.lower()

        self.docfile[self.documentCol] = self.docfile[self.documentCol].apply(lambda x: tokenizer.tokenize(x))

        apostophes = {"s":"is", "re":"are", "t":"not", "ve":"have", "ll":"will", "y":"you", "d":"would"}
        self.docfile[self.documentCol] = self.docfile[self.documentCol].apply(
            lambda x: [apostophes[w] if w in apostophes else w for w in x])

        # remove url
        self.docfile[self.documentCol] = self.docfile[self.documentCol].apply(
            lambda x: [w for w in x if not "https://" in w and not "http://" in w])

        # spell correction
        self.docfile[self.documentCol] = self.docfile[self.documentCol].apply(lambda x: [spell(w) for w in x])

        # stop words filtering
        stop_words = set(stopwords.words('english'))
        self.docfile[self.documentCol] = self.docfile[self.documentCol].apply(
            lambda x: [w for w in x if not w in stop_words])

        print(self.docfile[self.documentCol][0:5])
        # write to csv
        self.docfile.to_csv("processed_tweets.csv")

    def buildCorpus(self):
        # creates the corpus of words, and a document matrix
        # for testing - documentCol = "content"
        wordFrame = self.docfile[self.documentCol].str.decode('utf-8').str.cat(sep=' ')
        tokens = nltk.tokenize.word_tokenize(wordFrame)
        corpus = nltk.FreqDist(tokens)

        corpusFreq = nltk.FreqDist(corpus)

        tokens[0:10]

        wordFrame = self.docfile[self.documentCol].str.lower().str.split()
        wordFrame[1]

        corpus = pandas.Series(wordFrame).value_counts()

        cWords = collections.Counter()
        wordFrame.apply(cWords.update)
        pandas.DataFrame(wordList).stack().value_counts().head(10)
        wordList.head(10).value_counts()

        wordFrame2 = pandas.DataFrame(wordFrame, columns=['docs'])
        wordFrame2.head(10)
        wordFrame.apply(lambda x: pandas.value_counts(x.split(" "))).sum(axis=0)

        docfile2 = docfile

        docfile2[documentCol].apply(lambda x: pandas.value_counts(x.split(" "))).sum(axis=0)

        wordFrame['docs'].head(10)


class modelCollection(object):
# class to hold the models
    a = 1

class plsaModel(object):
# class for the plsa model
    a = 1

p = InputFile("tweets.csv", "content", "", "")
p.prepareDocs()

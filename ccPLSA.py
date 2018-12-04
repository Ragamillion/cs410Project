# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:59:26 2018

@author: Tom
"""
import numpy
import pandas
import nltk
import collections
from collections import Counter
import time
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')

class InputFile(object):
    #class for the input file and related processing
    filepath = ""
    
    def __Init__(self, filepath, documentCol, contextDiv, stopwords, nTopics, backgroundLambda):
        #initialize the object. Here filepath is the path to the data file, documentCol is the column in which the documents
        #are found, contextDiv is an array of the columns that determine the contexts, and stopwords is a .txt file containing
        #the stopwords
        self.filepath = filepath
        self.documentCol = documentCol
        self.contextDiv = contextDiv
        self.docFile = pandas.read_csv(self.filepath)
        self.stopwords = stopwords
        self.nTopics = nTopics
        self.backgroundLambda = backgroundLambda
        
    def buildContext(self):
        #builds the list of contexts, and creates a new column in the dataframe that contains the combined context for each
        #row. If there are any combinations of contexts that do not exist in the data, they will not be found in the list.
        
        
        filepath = "filteredtweets4.csv" #for testing purposes
        docfile = pandas.read_csv(filepath)
        
        docfile["plsaContext"] = ""
        contextDiv = pandas.Series(["account_category","dateSent"])
        
        for context in contextDiv:
            docfile["plsaContext"] = docfile["plsaContext"] + "|" + docfile[context] 
        
        docfile["plsaContext"] = docfile["plsaContext"].str[1:]
        contextList = docfile["plsaContext"].unique()
        
        
        
    def prepareDocs(self):
        #strips unwanted characters, prepares docs for analysis 
    
    def buildCorpus(self):
        #creates the corpus of words, and a document matrix
        documentCol = "content" #for testing - 
        
        '''
        Tried to create a matrix using pandas dataframes, NLTK tokens, and numpy arrays,
        but all were too slow.
        
        #NLTK solution
        wordFrame = docfile[documentCol].str.decode('utf-8').str.lower().str.cat(sep=' ')
        #docfile['documentTokens'] = docfile.apply(lambda row: 
        #       docfile[documentCol].str.decode('utf-8').str.lower(), axis=1)
         
        documentTokens = [ nltk.tokenize.casual.casual_tokenize(i.lower()) for i in docfile[documentCol]]
        
        tokens = nltk.tokenize.casual.casual_tokenize(wordFrame)
        corpus = nltk.FreqDist(tokens)
        cdict = dict(corpus)
        
        cFrame = pandas.DataFrame(corpus.most_common(), columns=['Word', 'Frequency'])
        
        #build document matrix
                

        docMatric = numpy.zeros(shape=(len(docfile.index), len(cFrame.index)), dtype=numpy.int8)
        #docMatrix = pandas.SparseDataFrame(index=docfile.index, columns= xRange( dtype=numpy.int8)
        
        for num, doc in enumerate(documentTokens, start=0):
            for word in doc:
                docMatric[num][cFrame.index[cFrame['Word']==word]] += 1
        
        '''
        #using sklearn data types 
        
        docVectors = CountVectorizer()
        
        docMat = docVectors.fit_transform(docfile[documentCol])
        
        fns = docVectors.get_feature_names()
        
        
        '''
        ccounts = corpus.pformat(10)
        ccounts = corpus.r_Nr()
        test = pandas.DataFrame(ccounts, columns = ['test'])
        #corpusFreq = nltk.FreqDist(corpus)
        
        tokens[0:10]
        
        wordFrame = docfile[documentCol].str.lower().str.split()
        wordFrame[1]
        
        corpus = pandas.Series(wordFrame).value_counts()
        
        cWords = collections.Counter()
        wordFrame.apply(cWords.update)
        pandas.DataFrame(wordList).stack().value_counts().head(10)
        wordList.head(10).value_counts()
        
        wordFrame2=pandas.DataFrame(wordFrame,columns=['docs'])
        wordFrame2.head(10)
        wfsums = wordFrame.apply(lambda x: pandas.value_counts(x.split(" "))).sum(axis = 0)
        
        #docfile2 = docfile
        
        #docfile2sums = docfile2[documentCol].apply(lambda x: pandas.value_counts(x.split(" "))).sum(axis = 0)

        #wordFrame['docs'].head(10)     
        
        #Counter solution
        wordFrame = docfile[documentCol]
        wc = collections.Counter(" ".join(wordFrame).split(" "))

        wc.len()
        '''
        
        # background dataset is taken from http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
        #Used in the paper by Go, A., Bhayani, R. and Huang, L., 2009. 
        #"Twitter sentiment classification using distant supervision". CS224N Project Report, Stanford, 1(12).
        backgroundDocs = pandas.read_csv('background tweets.csv')
        
        backgroundTokens = nltk.tokenize.casual.casual_tokenize(backgroundDocs['text'].str.decode('utf-8').str.lower().str.cat(sep=' '))
        
        bfreq = nltk.FreqDist(backgroundTokens)
        
        #bfreq.freq('the')
        
        
    def getTopics(self)
        
        nTopics = 4 #for testing purposes
        backgroundLambda = .2 #for testing purposes
        
        #build random arrays to initiate
        pDocTopic = numpy.random.random(size = (len(docfile), nTopics))
        pWordTopic = numpy.random.random(size = (len(fns), nTopics))
        pWordDocTopic = numpy.random.random(size = (len(docfile),len(fns), nTopics))
        
        #normalize so that sum of p = 1
        for doc in range(pDocTopic.shape[0]):
            nm = numpy.linalg.norm(pDocTopic[doc], ord=1)
            for tp in range(nTopics):
                pDocTopic[doc, tp] /= nm
        
        for wd in range(pWordTopic.shape[0]):
            nm = numpy.linalg.norm(pWordTopic[wd], ord=1)
            for tp in range(nTopics):
                pWordTopic[wd, tp] /= nm
        
        for n in range(pDocTopic.shape[0]):
            for x in docMat[n].nonzero()[1]:
                probVec = pDocTopic[n] * pWordTopic[x]
                nm = numpy.linalg.norm(probVec, ord=1)
                pWordDocTopic[n][x] = probVec/nm
        
        for z in range(nTopics):
            for n in range(pDocTopic.shape[0]):
                for x in docMat[n].nonzero()[1]:
                    s = 0
                    count = docMat[n,x]
                    s = s + count * pWordDocTopic[n][x][z]
                pWordTopic[x][z] = s
            nm = numpy.linalg.norm(pWordTopic[z], ord=1)
            pWordTopic[z] = pWordTopic[z]/nm
        
        for d_index in range(len(self.documents)):
                for z in range(number_of_topics):
                    s = 0
                    for w_index in range(vocabulary_size):
                        count = term_doc_matrix[d_index][w_index]
                        s = s + count * self.topic_prob[d_index, w_index, z]
                    self.document_topic_prob[d_index][z] = s
#                print self.document_topic_prob[d_index]
#                assert(sum(self.document_topic_prob[d_index]) != 0)
            normalize(self.document_topic_prob[d_index])
        
        for iteration in range(10):
            print "Iteration #" + str(iteration + 1) + "..."
            print "E step:"
            for d_index, document in enumerate(self.documents):
                for w_index in range(vocabulary_size):
                    prob = self.document_topic_prob[d_index, :] * self.topic_word_prob[:, w_index]
                    normalize(prob)
                    self.topic_prob[d_index][w_index] = prob
            print "M step:"
            # update P(w | z)
            for z in range(number_of_topics):
                for w_index in range(vocabulary_size):
                    s = 0
                    for d_index in range(len(self.documents)):
                        count = term_doc_matrix[d_index][w_index]
                        s = s + count * self.topic_prob[d_index, w_index, z]
                    self.topic_word_prob[z][w_index] = s
                normalize(self.topic_word_prob[z])
            
            # update P(z | d)
            for d_index in range(len(self.documents)):
                for z in range(number_of_topics):
                    s = 0
                    for w_index in range(vocabulary_size):
                        count = term_doc_matrix[d_index][w_index]
                        s = s + count * self.topic_prob[d_index, w_index, z]
                    self.document_topic_prob[d_index][z] = s
#                print self.document_topic_prob[d_index]
#                assert(sum(self.document_topic_prob[d_index]) != 0)
            normalize(self.document_topic_prob[d_index])
    
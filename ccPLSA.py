# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:59:26 2018
@author: Tom
@author: Akhila
"""
import numpy
import pandas
#import nltk
#import collections
#from collections import Counter
#import time
#import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from autocorrect import spell
from html2text import unescape

class InputFile(object):
    # class for the input file and related processing
    filepath = ""

    
    def __Init__(self, filepath, documentCol, contextDiv, stopwords, nTopics, backgroundLambda):
        #initialize the object. Here filepath is the path to the data file, documentCol is the column in which the documents
        #are found, contextDiv is an array of the columns that determine the contexts, and stopwords is a .txt file containing
        #the stopwords

        contextList = ""

    def __init__(self, filepath, documentCol, contextDiv, stopwords, nTopics):
        # initialize the object. Here filepath is the path to the data file, documentCol is the column in which the documents
        # are found, contextDiv is an array of the columns that determine the contexts, and stopwords is a .txt file containing
        # the stopwords

        self.filepath = filepath
        self.documentCol = documentCol
        self.contextDiv = contextDiv
        self.docfile = pandas.read_csv(self.filepath)
        self.stopwords = stopwords

        self.nTopics = nTopics
        #self.backgroundLambda = backgroundLambda
        
    def buildContext(self):
        #builds the list of contexts, and creates a new column in the dataframe that contains the combined context for each
        #row. If there are any combinations of contexts that do not exist in the data, they will not be found in the list.
        
        
        filepath = "filteredtweets4.csv" #for testing purposes
        docfile = pandas.read_csv(filepath)
        
        docfile["plsaContext"] = ""
        contextDiv = pandas.Series(["account_category","dateSent"])
        
        #concatenates the values in all the specified context columns for each row/document
        for context in contextDiv:
            docfile["plsaContext"] = docfile["plsaContext"] + "|" + docfile[context] 
        
        #remove leading "|"
        docfile["plsaContext"] = docfile["plsaContext"].str[1:]
        
        #gets list of unique contexts
        contextList = docfile["plsaContext"].unique().tolist()
        
        #Adds column for the index number of the context, for use in future looping
        docfile["contextNum"] = docfile["plsaContext"].apply(lambda x: contextList.index(x))
        
        #Add on the "Common" context that will be used in conjunction with the separate contexts
        contextList.append("Common")


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
        #creates the corpus of words, and a document matrix
        documentCol = "content" #for testing - 
        p = InputFile("tweets.csv", "content", "", "")
        p.prepareDocs()
        '''
        Tried to create a matrix using pandas dataframes, NLTK tokens, and numpy arrays,
        but all were too slow.

        '''
        #using sklearn data types. The docMat is a sparse matrix of word counts by document.
        #The fns is a unique list of words.
        
        docVectors = CountVectorizer()
        
        docMat = docVectors.fit_transform(docfile[documentCol])
        
        fns = docVectors.get_feature_names()
        
        
        '''

        '''
        
        # background dataset is taken from http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
        #Used in the paper by Go, A., Bhayani, R. and Huang, L., 2009. 
        #"Twitter sentiment classification using distant supervision". CS224N Project Report, Stanford, 1(12).
        #backgroundDocs = pandas.read_csv('background tweets.csv')
        
        #backgroundTokens = nltk.tokenize.casual.casual_tokenize(backgroundDocs['text'].str.decode('utf-8').str.lower().str.cat(sep=' '))
        
        #bfreq = nltk.FreqDist(backgroundTokens)
        
        #bfreq.freq('the')
        
        
    def getTopics(self):
        
        
             
        nTopics = 4 #for testing purposes
        #backgroundLambda = .2 #for testing purposes
        collectionLambda = .25
        nccollectionLambda = .75
        commonIndex = contextList.index("Common")
        
        #build random arrays to initiate
        pDocTopic = numpy.random.random(size = (len(docfile), nTopics))
        pWordTopic = numpy.random.random(size = (len(fns), len(contextList), nTopics))
        pWordDocTopic = numpy.random.random(size = (len(docfile),len(fns), len(contextList), nTopics))
        
        #normalize so that sum of p = 1
        for doc in range(pDocTopic.shape[0]):
            nm = numpy.linalg.norm(pDocTopic[doc], ord=1)
            for tp in range(nTopics):
                pDocTopic[doc, tp] /= nm
        
        for wd in range(pWordTopic.shape[0]):
            for cn in range(len(contextList)):
                nm = numpy.linalg.norm(pWordTopic[wd][cn], ord=1)
                for tp in range(nTopics):
                    pWordTopic[wd, cn, tp] /= nm		
		
        #E-Step: n is doc, x is word
        #This calculates the probabilities for the "hidden Z variable" for the 
        #specific context and the Common context. Results are then used in the
        #M-Step.
        for n in range(pDocTopic.shape[0]):
            for x in docMat[n].nonzero()[1]:
                step1 = (collectionLambda * pWordTopic[x][commonIndex] 
                    + (nccollectionLambda * pWordTopic[x][docfile["contextNum"][n]]))
                probVec = pDocTopic[n] * step1
                nm = numpy.linalg.norm(probVec, ord=1)
                pWordDocTopic[n][x][docfile["contextNum"][n]] = probVec/nm
                pWordDocTopic[n][x][commonIndex] = (collectionLambda * pWordTopic[x][commonIndex])/step1
        
        #M-Step: z is topic, n is doc, x is word. S is the specific context, t is the Common context
        #this calculates new probabilites for the individual words in each topic.
        for z in range(nTopics):
            for n in range(pDocTopic.shape[0]):
                s = 0
                t = 0
                for x in docMat[n].nonzero()[1]:
                    count = docMat[n,x]
                    s = s + count * pWordDocTopic[n][x][docfile["contextNum"][n]][z] * (1-pWordDocTopic[n][x][commonIndex][z])
                    t = t + count * pWordDocTopic[n][x][commonIndex][z] * pWordDocTopic[n][x][docfile["contextNum"][n]][z]
                pWordTopic[x][docfile["contextNum"][n]][z] = s
                pWordTopic[x][commonIndex][z] = t
            nm = numpy.linalg.norm(pWordTopic[:,:,z], ord=1)
            pWordTopic[:,:,z] = pWordTopic[:,:,z]/nm
        
        #Second part of the M-Step: n is doc, z is topic. s is to calculate the 
        #document-specific mixing weight.
        for n in range(pDocTopic.shape[0]):
            for z in range(nTopics):
                s = 0
                for x in docMat[n].nonzero()[1]:
                    count = docMat[n,x]
                    s = s + count * pWordDocTopic[n][x][z][docfile["contextNum"][n]]
                pDocTopic[n][z] = s
            nm = numpy.linalg.norm(pDocTopic[z], ord=1)
            pDocTopic[z] = pDocTopic[z]/nm
       
        
        #To print top 10 words in all contexts/topics
        for z in range(nTopics):
            print("Topic " + str(z))
            for c in range(len(contextList)):
                print("context is " + contextList[c])
                slist = sorted(range(len(pWordTopic)), key=lambda i: pWordTopic[i,c,z], reverse=True)[:10]
                for word in slist:
                    print(fns[word])
          

        #wordFrame.apply(lambda x: pandas.value_counts(x.split(" "))).sum(axis=0)

        #docfile2 = docfile

        #docfile2[documentCol].apply(lambda x: pandas.value_counts(x.split(" "))).sum(axis=0)

        #wordFrame['docs'].head(10)




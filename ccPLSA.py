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
from textblob import TextBlob 
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
#from autocorrect import spell
from html2text import unescape
import scipy.special
import gc

class ccMix(object):
    # class for the input file and related processing
    filepath = ""

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
        
        
        #filepath = "filteredtweets4.csv" #for testing purposes
        #docfile = pandas.read_csv(filepath)
        
        self.docfile["plsaContext"] = ""
        #self.contextDiv = pandas.Series(["account_category","dateSent"])
        
        #concatenates the values in all the specified context columns for each row/document
        for context in self.contextDiv:
            self.docfile["plsaContext"] = self.docfile["plsaContext"] + "|" + self.docfile[context] 
        
        #remove leading "|"
        self.docfile["plsaContext"] = self.docfile["plsaContext"].str[1:]
        
        #gets list of unique contexts
        self.contextList = self.docfile["plsaContext"].unique().tolist()
        
        #Adds column for the index number of the context, for use in future looping
        self.docfile["contextNum"] = self.docfile["plsaContext"].apply(lambda x: self.contextList.index(x))
        
        #Add on the "Common" context that will be used in conjunction with the separate contexts
        self.contextList.append("Common")


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
        #self.docfile[self.documentCol] = self.docfile[self.documentCol].apply(lambda x: [spell(w) for w in x])

        # stop words filtering
        stop_words = set(stopwords.words('english'))
        self.docfile[self.documentCol] = self.docfile[self.documentCol].apply(
            lambda x: [w for w in x if not w in stop_words])
        self.docfile[self.documentCol] = self.docfile[self.documentCol].apply(
            lambda x: ' '.join(x))
        print(self.docfile[self.documentCol][0:5])
        # write to csv
        #self.docfile.to_csv("processed_tweets.csv")

    def sentimentAnalysis(self):
        self.docfile["sentiment"] = ""
        self.docfile["sentiment"] = self.docfile[self.documentCol].apply(lambda x:  TextBlob(x).sentiment.polarity)
        self.docfile["sentiment"] = self.docfile["sentiment"].apply(lambda x:  "positive" if x > 0 else "neutral" if x == 0 else "negative")
        self.docfile["plsaContext"] = self.docfile["plsaContext"] + "|" + self.docfile["sentiment"]
        self.contextList = self.docfile["plsaContext"].unique().tolist()
        self.docfile["contextNum"] = self.docfile["plsaContext"].apply(lambda x: self.contextList.index(x))
        self.contextList.append("Common")
	print(self.docfile["sentiment"][0:5])

    def buildCorpus(self):
        #creates the corpus of words, and a document matrix
        #documentCol = "content" #for testing - 
        #p = InputFile("tweets.csv", "content", "", "")
        #p.prepareDocs()
        '''
        Tried to create a matrix using pandas dataframes, NLTK tokens, and numpy arrays,
        but all were too slow.

        '''
        #using sklearn data types. The docMat is a sparse matrix of word counts by document.
        #The fns is a unique list of words.
        
        self.docVectors = CountVectorizer()
        
        self.docMat = self.docVectors.fit_transform(self.docfile[self.documentCol])
        
        self.fns = self.docVectors.get_feature_names()
        
        
        '''

        '''
        
        #backgroundTokens = nltk.tokenize.casual.casual_tokenize(self.docFile[self.documentCol].str.decode('utf-8').str.lower().str.cat(sep=' '))
        
        #self.bfreq = nltk.FreqDist(backgroundTokens)
              
        #bfreq.freq('the')
        
        
    def getTopics(self, iterations):
        
        
        self.iterations = iterations   
        #nTopics = 4 #for testing purposes
        #backgroundLambda = .9
        collectionLambda = .25
        nccollectionLambda = .75
        commonIndex = self.contextList.index("Common")
        ndocs = len(self.docfile)
        nwords = len(self.fns)
        ncontext = len(self.contextList)
        
        #build random arrays to initiate - for pWordDocTopic the third dimension
        #is the context - index 0 for the specific context and index 1 for the Common
        pDocTopic = numpy.random.random(size = (ndocs, self.nTopics))
        self.pWordTopic = numpy.random.random(size = (nwords, ncontext, self.nTopics))
        pWordDocTopic = numpy.random.random(size = (ndocs, nwords, 2, self.nTopics))
        
        #normalize so that sum of p = 1
        for doc in range(pDocTopic.shape[0]):
            nm = sum(pDocTopic[doc])
            for tp in range(self.nTopics):
                pDocTopic[doc, tp] /= nm
        
        for wd in range(self.pWordTopic.shape[0]):
            for cn in range(ncontext):
                nm = sum(self.pWordTopic[wd][cn])
                for tp in range(self.nTopics):
                    self.pWordTopic[wd, cn, tp] /= nm		
        gc.collect()
        
        #set to log values to avoid underflow
        pDocTopic = numpy.log(pDocTopic)
        self.pWordTopic = numpy.log(self.pWordTopic)
        gc.collect()
        pWordDocTopic = numpy.log(pWordDocTopic)
        collectionLambda = numpy.log(collectionLambda)
        nccollectionLambda = numpy.log(nccollectionLambda)
        #backgroundLambda = numpy.log(backgroundLambda)
        
             
        gc.collect()
        for nIt in range(iterations):
            print("Iteration number: " + str(nIt+1))
            print("E-step")
            #E-Step: n is doc, x is word
            #This calculates the probabilities for the "hidden Z variable" for the 
            #specific context and the Common context. Results are then used in
            #the M-Step.
            for n in range(ndocs):
                for x in self.docMat[n].nonzero()[1]:
                    step1 = numpy.logaddexp((collectionLambda + self.pWordTopic[x][commonIndex]) 
                        , (nccollectionLambda + self.pWordTopic[x][self.docfile["contextNum"][n]]))
                    probVec = pDocTopic[n] + step1
                    nm = scipy.special.logsumexp(probVec)
                    pWordDocTopic[n][x][0] = probVec - nm
                    pWordDocTopic[n][x][1] = (collectionLambda + self.pWordTopic[x][commonIndex]) - step1
            
            print("M-step")
            #M-Step: z is topic, n is doc, x is word. S is the specific context, t is the Common context
            #this calculates new probabilites for the individual words in each topic.
            for z in range(self.nTopics):
                for n in range(ndocs):
                    s = 0
                    t = 0
                    for x in self.docMat[n].nonzero()[1]:
                        lcount = numpy.log(self.docMat[n,x])
                        with numpy.errstate(divide='ignore'): s = numpy.logaddexp(s , lcount + pWordDocTopic[n][x][0][z] + numpy.log((1-numpy.exp(pWordDocTopic[n][x][1][z]))))
                        with numpy.errstate(divide='ignore'): t = numpy.logaddexp(t , lcount + pWordDocTopic[n][x][1][z] + pWordDocTopic[n][x][0][z])
                    self.pWordTopic[x][self.docfile["contextNum"][n]][z] = s
                    self.pWordTopic[x][commonIndex][z] = t
                with numpy.errstate(divide='ignore'): nm = scipy.special.logsumexp(self.pWordTopic[:,:,z])
                self.pWordTopic[:,:,z] = self.pWordTopic[:,:,z]-nm
            
            #Second part of the M-Step: n is doc, z is topic. s is to calculate the 
            #document-specific mixing weight.
            for n in range(ndocs):
                for z in range(self.nTopics):
                    s = 0
                    for x in self.docMat[n].nonzero()[1]:
                        lcount = numpy.log(self.docMat[n,x])
                        with numpy.errstate(divide='ignore'): s = numpy.logaddexp(s , lcount + pWordDocTopic[n][x][0][z])
                    pDocTopic[n][z] = s
                with numpy.errstate(divide='ignore'): nm = scipy.special.logsumexp(pDocTopic[z])
                pDocTopic[z] = pDocTopic[z] - nm
         
          
    def printTopics(self,n):  
        #To print top n words in all contexts/topics
        for z in range(self.nTopics):
            print("Topic " + str(z))
            for c in range(len(self.contextList)):
                print("context is " + self.contextList[c])
                slist = sorted(range(len(self.pWordTopic)), key=lambda i: self.pWordTopic[i,c,z], reverse=True)[:n]
                for word in slist:
                    print(self.fns[word])
          


test1 = ccMix("filteredtweets4.csv", "content", ["account_category","dateSent"],"stopwords.txt", 4)
test1.buildContext()
test1.prepareDocs()
test1.buildCorpus()
test1.sentimentAnalysis()
#a1 = time.time()
test1.getTopics(10)
#print(time.time()-a1)
test1.printTopics(5)


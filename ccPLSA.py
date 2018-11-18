# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:59:26 2018

@author: Tom
"""
import numpy
import pandas
import os

class InputFile(object):
    #class for the input file and related processing
    filepath = ""
    
    def __Init__(self, filepath, documentCol, contextDiv, stopwords):
        #initialize the object. Here filepath is the path to the data file, documentCol is the column in which the documents
        #are found, contextDiv is an array of the columns that determine the contexts, and stopwords is a .txt file containing
        #the stopwords
        self.filepath = filepath
        self.documentCol = documentCol
        self.contextDiv = contextDiv
        self.docFile = pandas.read_csv(self.filepath)
        self.stopwords = stopwords
        
    def buildContext(self):
        #builds the list of contexts, and creates a new column in the dataframe that contains the combined context for each
        #row. If there are any combinations of contexts that do not exist in the data, they will not be found in the list.
        
        filepath = "Test values - Sheet1.csv" #for testing purposes
        #docfile = pandas.read_csv(filepath)
        contextDiv = pandas.Series(["col1","col2","col4"])
        docfile["plsaContext"] = ""
        
        for context in contextDiv:
            docfile["plsaContext"] = docfile["plsaContext"] + "|" + docfile[context] 
        
        docfile["plsaContext"] = docfile["plsaContext"].str[1:]
        contextList = docfile["plsaContext"].unique()
        
    def prepareDocs(self):
        #strips unwanted characters, prepares docs for analysis 
    
    def buildCorpus(self):
        #creates the corpus of words, and a document matrix
        wordList = docfile[documentCol].str.split().stack().value_counts()

class modelCollection(object):
    #class to hold the models

class plsaModel(object):
    #class for the plsa model
        
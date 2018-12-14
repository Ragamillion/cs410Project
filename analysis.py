import numpy
import pandas
import json

class analysis(object):
  def __init__(self, pDocTopicFile, pWordTopicFile, contextFile, fnsFile, topTopics, topWordsPerTopic):
    self.pDocTopic = numpy.load(pDocTopicFile)
    self.pWordTopic = numpy.load(pWordTopicFile)
    self.context = pandas.read_csv(contextFile)
    with open(fnsFile, 'r') as filehandle:  
      self.fns = json.load(filehandle)
    self.topTopics = topTopics
    self.topWordsPerTopic = topWordsPerTopic
    self.contextList = []
  
  def prepareResults(self):
    self.contextList = self.context["plsaContext"].unique().tolist()
    self.context["contextNum"] = self.context["plsaContext"].apply(lambda x: self.contextList.index(x))
    
    maxTopicsPerContext = numpy.zeros((len(self.contextList), len(self.pDocTopic[0])))
    for i in range(len(self.context["contextNum"])):
      idx = self.context["contextNum"][i]
      maxTopicsPerContext[idx] = maxTopicsPerContext[idx] + self.pDocTopic[i]
    #print(maxTopicsPerContext)
    #print(self.pDocTopic)
    
    contextWordList = []
    self.result = pandas.DataFrame()
    self.result["context"] = ""
    self.result["most tweeted topics"] = ""
    self.result["context"] = self.contextList
    for i in range(len(self.contextList)):
      sList = sorted(range(len(maxTopicsPerContext[i])), key=lambda j: maxTopicsPerContext[i,j], reverse=True)[:self.topTopics]
      topicWordList = []
      for j in sList:
        indexList = sorted(range(len(self.pWordTopic)), key=lambda k: self.pWordTopic[k,i,j], reverse=True)[:self.topWordsPerTopic]
        wordList = []
        for idx in indexList:
          wordList.append(self.fns[idx])
        topicWordList.append(wordList)
      contextWordList.append(topicWordList)
    self.result["most tweeted topics"] = contextWordList
    self.result.to_csv("result.csv", encoding='utf8')
    #print(contextWordList)

p = analysis("pDocTopic.npy", "pWordTopic.npy", "context.csv", "fns.txt", 4, 5)
p.prepareResults()

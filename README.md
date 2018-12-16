# cs410Project
Final project for cs410  
Tom Phelan and Akhila Prodduturi  

This is a context based topic analysis tool for use with collections of tweets, written in Python.
The tool has been developed using the cc-Mix model proposed in the paper "A cross-collection
mixture model for comparative text mining" by ChengXiang Zhai, Atulya Velivelli, and Bei Yu, 2004.
The goal is to allow the user to provide a csv file with the documents in one column and the contextual
variables in a series of other columns in the dataset. The tool then allows the user to clean up
the documents and analyse them. It produces a user-specified number of topic models and word distributions for those topics.

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
buildContext() - required - Must be run first. Takes the list of context columns provided by the user
    and combines the values within those columns to create a single context for each row.  
prepareDocs() - optional - performs document clean-up and preperation for analysis, including
    removing stop words. Focused on tweet documents.  
sentimentAnalysis() - optional - Performs sentiment analysis on each document and adds on the
    resulting sentiment to the document's context. Sentiment values are "Positive", "Negative",
    and "Neutral".  
buildCorpus() - required - Creates the matrix of word counts for the document collection.  
getTopics(iterations) - required - Creates the topic models and performs the analysis. The iterations  
    parameter controls how many EM iterations are performed. We recommend doing at least 10.
printTopics(n) - optional - Prints the top n words for each context and model by probability.  

An example of using this tool can be found in the code below, using the provided .csv file:  

test1 = ccMix("filteredtweets4.csv", "content", ["account_category","dateSent"],"stopwords.txt", 4)  
test1.buildContext()  
test1.prepareDocs()  
test1.buildCorpus()  
test1.sentimentAnalysis()  
test1.getTopics(25)  
test1.printTopics(10)  

analysis.py is added to this tool to present the results in a tabular format. In this scenario we've chosen to display top 4 topics for each context and top 5 most tweeted topic words for those topics.



Future development:  
Currently the tool implements the cc-Mix model without the proposed Background component,  
and instead relys on removing common words. A future implementation could add in this feature.  
  
Dr. Zhai has mentioned that it is possible to combine the E and M steps into a single loop through  
the corpus, which would remove the necessity of storing the Document-Word-Topic probabilities.   
This very large matrix is a big memory drain and so future work on this tool could revamp the  
algorithm to free up memory.  

Twitter documents are rarely long enough to have a topic word multiple times. As a result,  
it would be interesting to explore a binary tracking of word presence instead of word count.  
This could free up memory and reduce computation time if implemented well.  

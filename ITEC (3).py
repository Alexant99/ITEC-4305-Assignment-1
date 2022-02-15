#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install --upgrade category_encoders
#pip install requests
#pip install --upgrade category_encoders
#pip install bs4
#pip install gensim
#conda install -c conda-forge lightgbm 


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from IPython.display import Image
from sklearn import tree
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
import os.path
from sklearn.datasets import load_iris
from lightgbm import LGBMRegressor
from category_encoders import OrdinalEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import requests
from bs4 import BeautifulSoup
from sklearn.naive_bayes import GaussianNB
import glob
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import operator
#jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10

#from gensim.parsing.preprocessing import remove_stopwords

#nltk.download('stopwords') #needs to run to download the stopwords from the nltk package
#nltk.download('punkt')#needs to run to download the stopwords from the nltk package


# In[3]:


lineNum = 0
lineNumber = 1
tokens_without_sw = []
tokens = []
testTokens = []
pureText = ''
linecheck = ''
stopwordCheck = ''
removedStopwords = ''
tokens_without_bl = []
testTokensWithoutBl = []
fileName = []
paragraphText = ''
inverted_index = {}
frequency_index = {}
testInverted_index = {}
testFrequency_index = {}
fileNumber = 1
filePath = {'course', 'faculty', 'student'}
stopword_list = nltk.corpus.stopwords.words('english')
vectorizer = CountVectorizer()
fileNum = 1
textExtracted = []
testParagraphText = []


# In[4]:


#created a blacklist and special character list to remove specific words or characters that will not add to the index
blacklist = ['-', '--', ':', '.', ',', '?', '<.*?>', '@', '['']','[]', '/', '...', '....', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '1.0', '2.0', '3.0', 'MIME-Version', 'Server', 'CERN/3.0','Date','Monday','06-Jan-97', '19:30:42', 'GMT', 'Content-Type:', 'text/html', 'Content-Length:', '1707', 'Last-Modified', 'Wednesday', '11-Dec-96', '21:39:13','GMT', '<!BODY BGCOLOR=#040404 text=#ffffff link=#44ffff vlink=#ffff00 alink=#ff2222>']
special_char=[",",":"," ",";",".","?", "=", "--", "|"]
special_char_tuple = (",",":"," ",";",".","?", "=", "--", '<!WA0><!WA0><!WA0>', '<!WA1><!WA1><!WA1>','<!WA2><!WA2><!WA2>', '<!WA3><!WA3><!WA3>','<!WA4><!WA4><!WA4>')
anotherone = ("<")


# In[5]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result: \n==========================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred):.4f}\n")
        print(f"Classification Report:  \n \tPrecision: {precision_score(y_train, pred)}\n \tRecall Score: {recall_score(y_train, pred)}")
        print(f"Confusion Matrix: {confusion_matrix(y_train, clf.predict(X_train))}")
    elif train == False:
        print("Test Result: \n==========================================")
        pred = clf.predict(X_test)
        print(f"Accuracy Score: {accuracy_score(y_test, pred):.4f}\n")
        print(f"Classification Report:  \n \tPrecision: {precision_score(y_test, pred)}\n \tRecall Score: {recall_score(y_test, pred)}")
        print(f"Confusion Matrix: {confusion_matrix(y_test, pred)}")


# In[6]:


#
x = []
for path in filePath:
    for file_name in os.listdir(('C:/Users/alexa/Documents/ITEC4305A1/train/' + path)):
        fileNumber += 1
        with open(os.path.join(('C:/Users/alexa/Documents/ITEC4305A1/train/' + path), file_name)) as html_file:
            fileName.append(file_name)
            soup = BeautifulSoup(html_file, "lxml").text
            textExtracted.append(soup)


# In[7]:


#Outputs each file in the directory
print(fileName)


# In[8]:


#
for line in textExtracted:
    if line not in blacklist and line not in stopword_list:
        tokens_without_bl.append(line) 


# In[9]:


#
for page in tokens_without_bl:
    print(page)
    if page != [] and page != '' and page not in stopword_list and page not in blacklist:
            removedStopwords = page.split()
            print(removedStopwords)
            if removedStopwords != [] and removedStopwords != '' and removedStopwords not in stopword_list and removedStopwords not in blacklist:
                tokens_without_sw.append(removedStopwords)


# In[10]:


#
inverted_index = {}
a = 0
x = 0
for line in tokens_without_sw:
    for word in line:
        print(word)
        word = word.lower()
        if word not in stopword_list and word not in blacklist:
            if word not in special_char and not isinstance(word, (int, float)):
                if len(word) >= 3:
                    if word.endswith("ies"):
                        word.replace('ies', 'y', 1)
                    elif word.endswith("ed"):
                        word = word[0:-2]
                    elif word.endswith("ing"):
                        word = word[0:-3]
                    if word.endswith("..."):
                        word = word[0:-3]
                    if word.endswith(".") or word.endswith(",") or word.endswith(":"):
                        word = word[0:-1]
                    if word.startswith(special_char_tuple):
                        word = word[-1:0] 
                    if word.startswith("<!wa0><!wa0><!wa0><!wa0>"):
                        word = word[-22:0]
                    if word.startswith(anotherone):
                        word = word[21:0]
                    

                    if word not in inverted_index:
                        inverted_index[word] = ["Frequency:", fileNum]
                    if word in inverted_index:
                        inverted_index[word] = (fileNum)
        lineNumber += 1
    fileNum += 1


# In[11]:


#Outputs the inverted index
print(inverted_index)


# In[12]:


list(inverted_index.keys())


# In[13]:


print(len(inverted_index))


# In[14]:


x = len(inverted_index)
x


# In[15]:


#
testTextExtracted = []
testfileName = []
testTokensWithoutBl = []
testRemovedStopwords = []
testTokensWithoutSw = []
testFileNum = 1
testLineNum = 1
testInverted_index = {}

for path in filePath:
    for testFile_Name in os.listdir(('C:/Users/alexa/Documents/ITEC4305A1/test/' + path)):
        with open(os.path.join(('C:/Users/alexa/Documents/ITEC4305A1/test/' + path), testFile_Name)) as html_file:
            soup = BeautifulSoup(html_file, 'lxml').text
            testfileName.append(testFile_Name)
            testTextExtracted.append(soup)


# In[16]:


#
for textTest in testTextExtracted:
    if textTest not in blacklist:
        testTokensWithoutBl.append(textTest)


# In[17]:


#
for word in testTokensWithoutBl:
    for stopwordCheck in word:
        if stopwordCheck not in stopword_list:
            testRemovedStopwords = stopwordCheck.split()
            if testRemovedStopwords != []:
                testTokensWithoutSw.append(testRemovedStopwords)


# In[18]:


#
for line in testTokensWithoutSw:
    for word in line:
        word = word.lower()
        if word not in stopword_list and word not in blacklist:
            if word not in special_char and not isinstance(word, (int, float)):
                if len(word) >= 3:
                    if word.endswith("ies"):
                        word.replace('ies', 'y', 1)
                    elif word.endswith("ed"):
                        word = word[0:-2]
                    elif word.endswith("ing"):
                        word = word[0:-3]
                    elif word.endswith(".") or word.endswith(",") or word.endswith(":") or word.startswith("+") or word.startswith("-") or word.startswith("/") or word.startswith("*") or word.startswith("(") or word.endswith(")"):
                        word = word[0:-1]

                    if word not in testInverted_index:
                        wordDict.append(word)
                        testInverted_index[word] = ["frequency:", x]
                    if word in inverted_index:

                        testInverted_index[word].append(x)
        testLineNum += 1
    testFileNum += 1


# In[19]:


#Tranforms the inverted index list into a bag of words vector
#bagOfWords = vectorizer.fit_transform([inverted_index])


# In[30]:


df = pd.DataFrame()

df = df.from_dict(inverted_index, orient='index', columns = ['frequency'])
print(df)


# In[31]:


#vectorizer = CountVectorizer()
#Xtrain,yTrain=vectorizer.fit_transform(df['word']).toarray(),vectorizer.fit_transform(df['frequency']).toarray()


# In[32]:


#creates a target and another table that does not contain that column(s)
target = df.sample(n=3, axis='columns')
pf = df.drop(target, axis='columns')


# In[ ]:


#Decision Tree Classifier Testing and Training sets
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2)
clf = tree.DecisionTreeClassifier(max_depth = 5)
clf.fit(s,target)

#outputs the accuracy of the testing and training sets
print_score(clf, X_train,y_train, X_test, y_test, train=True)
print_score(clf, X_train,y_train, X_test, y_test, train=False)


# In[ ]:


#Naive Bayes Classifier Testing and Training sets
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
#outputs the accuracy of the testing and training sets
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


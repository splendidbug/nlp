"""
Created on Sun Mar 29 20:51:17 2020

@author: Lenovo
"""
import re # for regular expressions
import pandas as pd 
pd.set_option("display.max_colwidth", 200)
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk 
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv('newTraffic.csv')
test = pd.read_csv('newTest.csv')
train.head()

combi = train.append(test, ignore_index=True)


def remove_pattern(input_txt, pattern): #removes unwanted text pattern
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['text'], "@[\w]*") #removing handles (@name)

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") #removes punctuations, numbers, special characters
combi.head(5)

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) #removing short words upto 3 chars like hmmm, oh
 
#stopword removal
stop = stopwords.words('english')

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))

#stemming
from nltk.stem.porter import *
stemmer = PorterStemmer()

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split() ])) # stemming

#pos tagging
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: nltk.pos_tag(nltk.word_tokenize(x))) 


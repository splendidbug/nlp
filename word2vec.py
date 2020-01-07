                                    #word2vec
#Goal: Create woprd vectors from GOT dataset and analyse them to see semantic similarity

from __future__ import absolute_import, division, print_function #future has 2 underscores
#for word encoding
import codecs
#regex - searches a something in a large set
import glob
#concurrency - multi-threading
import multiprocessing 
#dealing with operating systems, like reading a file
import os
#pretty printing, human readable format
import pprint
#regular expression - I don't know what it does exactly, just import
import re
#natural language toolkit
import nltk
#word 2 vec
import gensim.models.word2vec as w2v
#dimension reduction as we're gnna have hundereds of dimensions
import sklearn.manifold
#math
import matplotlib.pyplot as plt
#parse pandas as pd
import pandas as pd
#visualization
import seaborn as sns
nltk.download("punkt") #used to tokenize the corpus
nltk.download("stopwords") #removes the stopwords like a, and, the, etc
book_filenames = sorted(glob.glob("got*.txt"))
corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()

#Splitting the corpus into sentences 
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #loads the downloaded "punkt" model in the memory
raw_sentences = tokenizer .tokenize(corpus_raw)
#converting into list of words
#removing unnecessary characters, splitting into words, no hyphens
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words
#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentences)>0:
        sentences.append(sentence_to_wordlist(raw_sentence))
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))
#ONCE we have vectors
#step 3 - build model
#3 main tasks that vectors help with
#DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
#more dimensions, more computationally expensive to train
#but also more accurate
#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words. 
#i.e., reducing the number of repeated words as they've already been converted to vectors
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1
#This is from the gensim library. We give any kind of corpus, it'll create a model, it'll train it.
#It'll gives us the words, how similar they are, what does'nt match, gives us vectors to use it later on if required

thrones2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
#Building the vocabulary
thrones2vec.build_vocab(sentences) 

thrones2vec.train(sentences, total_examples=thrones2vec.corpus_count, epochs=40)

#save to file, can be used later
if not os.path.exists("trained"):
    os.makedirs("trained")
    
thrones2vec.save(os.path.join("trained", "thrones2vec.w2v"))

#Exploring the trained model
thrones2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v")) #loading the saved file

#compressing the vectors into 2d space and plotting them
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = thrones2vec.wv.syn0

#training t-sne
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

#plotting 
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
            for word in thrones2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)
    
points.head(50)

sns.set_context("poster")

#plotting on the graph
points.plot.scatter("x", "y", s=10, figsize=(20, 12))

#zooming in
def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

plot_region(x_bounds=(0, 1), y_bounds=(1, 18.5))

#Exploring semantic similarities b/w book characters
thrones2vec.most_similar("woman")
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from gensim import corpora, models, similarities, matutils
# # logging for gensim (set to INFO)
# import logging
# logging.basicConfig(filename='gensim.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import matplotlib.pyplot as plt

# add additional stop words
additional_stop_words = ['like', 'said', 'would', 'could', 'should', 'one']
my_stop_words = stop_words.union(additional_stop_words)


# get counts and remove stop words
def vectorizeText(inputText, min_df=1, max_df=1.0):
    cv = CountVectorizer(stop_words=my_stop_words, min_df=min_df, max_df=max_df)
    X = cv.fit_transform(inputText)
    
    return X, cv

# get tf-idfs and remove stop words
def vectorizeTextIDF(inputText, min_df=1, max_df=1.0):
    cv_tfidf = TfidfVectorizer(stop_words=my_stop_words, min_df=min_df, max_df=max_df)
    X_tfidf = cv_tfidf.fit_transform(inputText)
    
    return X_tfidf, cv_tfidf

def doNMF(numTopics, vectorized_matrix, vectorizer):
    
    nmf_model = NMF(numTopics, random_state=84597)
    doc_topic_matrix = nmf_model.fit_transform(vectorized_matrix)

    topic_word_matrix = nmf_model.components_
    top_word_ids = topic_word_matrix.argsort(axis=1)[:,-1:-7:-1]

    words = vectorizer.get_feature_names()
    top_topic_words = [[words[word_id] for word_id in topicNum] for topicNum in top_word_ids]
    
    print('\nNMF Topic Words:')
    for i in range(numTopics):
        print("Topic %d:" % i, end='')
        for word in top_topic_words[i]:
            print(' %s' % word, end='')
        print()
        
    return doc_topic_matrix

    
def dokMeans(numClusters, vectorized_matrix, vectorizer):

    km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=10, random_state=425)
    doc_clusters = km.fit_predict(vectorized_matrix)

    order_centroids = km.cluster_centers_.argsort()[:,::-1]
    words = vectorizer.get_feature_names()

    print('\nkMeans Cluster Words:')
    for i in range(numClusters):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :6]:
            print(' %s' % words[ind], end='')
        print()
        
    return doc_clusters


def doLDA(numTopics, vectorized_matrix, vectorizer):
    
    doc_words = vectorized_matrix.transpose()
    corpus_for_lda = matutils.Sparse2Corpus(doc_words)
    
    id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())

    lda = models.LdaModel(corpus=corpus_for_lda, num_topics=numTopics, id2word=id2word, passes=5, random_state=67676)#, eval_every=1) # eval_every for convergence plotting
    
    print('\nLDA Topic Words:')
    for i in range(numTopics):
        print("Topic %d:" % i, end=' ')
        print(lda.print_topic(i, 6))
    
    lda_transformed_corpus = lda[corpus_for_lda]
    lda_doc_probs = [doc for doc in lda_transformed_corpus]
    
    return lda_doc_probs

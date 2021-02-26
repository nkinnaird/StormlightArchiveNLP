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
additional_stop_words = ['', 'like', 'said', 'would', 'could', 'should', 'room', 'one', 'two', 'three', 'four', 'five', 'woman', 'man', 'look', 'know', 'think', 'want', 'go', 'us', 
                         'thing', 'get', 'part']

default_stop_words = stop_words.union(additional_stop_words)

name_words = ['kaladin', 'kal', 'dalinar', 'adolin', 'shallan', 'venli', 'eshonai', 'raboniel', 'navani', 'teft', 'rock', 'rlain', 'szeth', 'taravangian', 'lirin', 'jasnah', 'veil',
              'pattern', 'syl', 'sylphrena', 'mraiz', 'kabsal', 'yalb', 'balat', 'jushu', 'wikim', 'helaran', 'malis', 'tyn', 'gaz', 'vathah', 'macob', 'tvlakv', 'bluth', 'tag',
              'ialai', 'amaram', 'eylita', 'ishnah', 'grund', 'elhokar', 'gavilar', 'mishim', 'iyatil', 'sebari', 'sadea', 'renarin', 'palona', 'tsa', 'wit', 'moash', 'dabbid', 'lopen',
              'leshwi', 'beryl', 'azur', 'roshon', 'tien', 'hesina', 'laral', 'zahel', 'sigzil', 'lamaril', 'skar', 'fen', 'aladar', 'tanalan', 'evi', 'yanagawn', 'mink', 'ishar',
              'nohadon', 'ruthar', 'kadash', 'noura', 'jaxlim', 'timbr', 'ulim', 'demid', 'nale', 'thude', 'jakamav', 'kelek', 'sekeir', 'notum', 'maya', 'godek', 'sibl', 'teoﬁl',
              'falilar', 'kalami', 'geranid', 'ashir', 'sja', 'anat', 'nin', 'rysn', 'chiri', 'vstim', 'talik', 'adrotagia', 'mrall', 'dukar', 'wyndl', 'lift', 'gawx', 'davim',
              'cenn', 'dallet', 'roion', 'nimi', 'hobber', 'lunamor', 'shumin', 'jesevan']


# get counts and remove stop words
def vectorizeText(inputText, min_df=1, max_df=1.0, dropNameWords=False):
    these_stop_words = default_stop_words
    if dropNameWords: these_stop_words = these_stop_words.union(name_words)
    
    cv = CountVectorizer(stop_words=these_stop_words, min_df=min_df, max_df=max_df)
    X = cv.fit_transform(inputText)
    
    return X, cv

# get tf-idfs and remove stop words
def vectorizeTextIDF(inputText, min_df=1, max_df=1.0, dropNameWords=False):
    these_stop_words = default_stop_words
    if dropNameWords: these_stop_words = these_stop_words.union(name_words) 
            
    cv_tfidf = TfidfVectorizer(stop_words=these_stop_words, min_df=min_df, max_df=max_df)
    X_tfidf = cv_tfidf.fit_transform(inputText)
    
    return X_tfidf, cv_tfidf


# make dictionaries from NMF topic word matrices for word cloud plotting
def getNMF_TopicWord_Dicts(topic_word_matrix, vectorizer):
    
    words = vectorizer.get_feature_names()

    list_of_dicts = []
    
    for t in topic_word_matrix:     
        topic_dict = dict()
        
        for word_id, value in enumerate(t):
            topic_dict[words[word_id]] = value
        
        list_of_dicts.append(topic_dict)
                        
    return list_of_dicts
    
    
def doNMF(numTopics, vectorized_matrix, vectorizer):
    
    nmf_model = NMF(numTopics, max_iter=500, random_state=84597)
    doc_topic_matrix = nmf_model.fit_transform(vectorized_matrix)

    topic_word_matrix = nmf_model.components_
    top_word_ids = topic_word_matrix.argsort(axis=1)[:,-1:-7:-1]

    words = vectorizer.get_feature_names()
    top_topic_words = [[words[word_id] for word_id in topicNum] for topicNum in top_word_ids]
    
    top_word_each_topic = []

    print('\nNMF Topic Words:')
    for i in range(numTopics):
        print("Topic %d:" % i, end='')
        for j, word in enumerate(top_topic_words[i]):
            print(' %s' % word, end='')
            if j == 0 : top_word_each_topic.append(word)
        print()
        
    return doc_topic_matrix, topic_word_matrix, top_word_each_topic

    
def dokMeans(numClusters, vectorized_matrix, vectorizer):

    km = KMeans(n_clusters=numClusters, init='k-means++', max_iter=100, n_init=10, random_state=425)
    doc_clusters = km.fit_predict(vectorized_matrix)

    order_centroids = km.cluster_centers_.argsort()[:,::-1]
    words = vectorizer.get_feature_names()
    
    top_word_each_cluster = []
        
    print('\nkMeans Cluster Words:')
    for i in range(numClusters):
        print("Cluster %d:" % i, end='')
        for j, ind in enumerate(order_centroids[i, :6]):
            print(' %s' % words[ind], end='')
            if j == 0: top_word_each_cluster.append(words[ind])
        print()
        
    return doc_clusters, top_word_each_cluster


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

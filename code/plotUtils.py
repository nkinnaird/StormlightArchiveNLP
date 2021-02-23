import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

names_to_colors = [('kaladin', "#1f77b4"), ('shallan', "#ff7f0e"), ('dalinar', "#2ca02c"), ('venli', "#d62728"), ('navani', "#9467bd"), 
                   ('adolin', "#8c564b"), ('szeth', "#e377c2"), ('veil', "#7f7f7f"), ('kal', "#bcbd22"), ('eshonai', "#17becf"),
                   ('taravangian', "#e377c2"), ('lirin', "#bcbd22")] # szeth-taravangian the same color, lirin-kal the same color
color_dict = dict(names_to_colors)


# use this function to make the colors consistent across various plots - colors will grab by index from the default colors
def getPalette(labels):
        
    grabbed_colors = [color_dict[label] for label in labels]    
    customPalette = sns.set_palette(sns.color_palette(grabbed_colors))
    return customPalette


# plot number of counts in each cluster against booknum
def getClusterCounts(df_with_results, top_words, bookNum):
    
    unique_cluster_values = df_with_results['kMeans'].sort_values().unique()
    labels = [top_words[item] for item in unique_cluster_values]
            
    fig, ax = plt.subplots()
    sns.countplot(x='kMeans', data=df_with_results.sort_values(by='kMeans'), palette=getPalette(labels)) # sort kMeans column by value in order to plot names and colors consistently
        
    ax.set_xticklabels(labels)
    
    if bookNum == 0:
        plt.title("All Books kMeans")
    else:
        plt.title(f"Book {bookNum} kMeans")

    plt.ylabel("Number of Chapters")
    plt.xlabel("Character Cluster")

    plt.show()
    
    
# plot number of counts in each topic (max value) against booknum
def getNMFCounts(df_with_results, top_words, bookNum):
    
    xs = [np.asarray(values).argmax() for values in df_with_results['NMF']]
    xs.sort()
    
    labels = [top_words[item] for item in set(xs)]

    fig, ax = plt.subplots()        
    sns.countplot(x=xs, palette=getPalette(labels))
        
    ax.set_xticklabels(labels)
    
    if bookNum == 0:
        plt.title("All Books NMF")
    else:
        plt.title(f"Book {bookNum} NMF")
        
    plt.ylabel("Number of Chapters")
    plt.xlabel("Character Topic")
            
    plt.show()
    

def makeMDSPlot(vectorized_matrix, df_with_results, top_words):

    distances = cosine_distances(vectorized_matrix)
    
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=232, max_iter=300, verbose=1)
    positions_2d = mds.fit_transform(distances)
#     print('Final stress value: %f' %mds.stress_)

    df_temp = pd.DataFrame(positions_2d, columns=['comp1', 'comp2'])
    df_temp['Cluster'] = df_with_results['kMeans']
    df_temp['Label'] = df_temp['Cluster'].apply(lambda row: top_words[row])    
    
    labels = [top_words[item] for item in list(df_temp['Cluster'].sort_values().unique())]
    
    sns.lmplot(x='comp1', y='comp2', data=df_temp.sort_values(by='Cluster'), hue='Label', palette=getPalette(labels), fit_reg=False)

    plt.show()
    
    
def makeTSNEPlot(vectorized_matrix, df_with_results, top_words, inputPerplexity=50):
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=inputPerplexity, n_iter=1000, learning_rate=200, random_state=6321)
    tsne_results = tsne.fit_transform(vectorized_matrix)
    
    df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])
    df_tsne['Cluster'] = df_with_results['kMeans']
    df_tsne['Label'] = df_tsne['Cluster'].apply(lambda row: top_words[row])

    labels = [top_words[item] for item in list(df_tsne['Cluster'].sort_values().unique())]
    
    print(labels)

    sns.lmplot(x='comp1', y='comp2', data=df_tsne.sort_values(by='Cluster'), hue='Label', palette=getPalette(labels), fit_reg=False)

    plt.show()
    
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


# def makeMDSPlot(vectorized_matrix):

#     distances  = cosine_distances(vectorized_matrix)
    
#     mds = MDS(n_components=2, dissimilarity="precomputed", random_state=232, max_iter=1000, verbose=1)
#     positions_2d = mds.fit_transform(distances)
# #     print('Final stress value: %f' %mds.stress_)

#     xs, ys = positions_2d[:, 0], positions_2d[:, 1]

#     fig, ax = plt.subplots(figsize=(9, 6))
#     ax.plot(xs, ys, marker='o', markersize=6, linestyle='', color='orange', alpha=1.0, mec="none" ) 
#     ax.set_aspect('equal')


#     # ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
#     # ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
#     # ax.set_xlim(-0.85,1.7)
#     # ax.set_ylim(-0.85,0.85)

#     plt.show()

def makeMDSPlot(vectorized_matrix, df_with_results):

    distances  = cosine_distances(vectorized_matrix)
    
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=232, max_iter=300, verbose=1)
    positions_2d = mds.fit_transform(distances)
#     print('Final stress value: %f' %mds.stress_)

    df_temp = pd.DataFrame(positions_2d, columns=['comp1', 'comp2'])
    df_temp['Cluster'] = df_with_results['kMeans']
    sns.lmplot(x='comp1', y='comp2', data=df_temp, hue='Cluster', fit_reg=False)

    plt.show()
    
    
def makeTSNEPlot(vectorized_matrix, df_with_results):
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=200, random_state=6321)
    tsne_results = tsne.fit_transform(vectorized_matrix)
    
    df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])
    df_tsne['Cluster'] = df_with_results['kMeans']
    sns.lmplot(x='comp1', y='comp2', data=df_tsne, hue='Cluster', fit_reg=False)

    plt.show()
    
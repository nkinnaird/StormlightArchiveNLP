import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from wordcloud import WordCloud

params = {'legend.fontsize': 14,
          'legend.title_fontsize': 14,
          'figure.figsize': (8, 6),
          'axes.labelsize': 16,
          'axes.titlesize': 18,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14}
plt.rcParams.update(params)


# connect names to colors for specific plotting
names_to_colors = [('kaladin', "#1f77b4"), ('shallan', "#ff7f0e"), ('dalinar', "#2ca02c"), ('venli', "#d62728"), ('navani', "#9467bd"), 
                   ('adolin', "#8c564b"), ('szeth', "#e377c2"), ('veil', "#7f7f7f"), ('kal', "#bcbd22"), ('eshonai', "#17becf"),
                   ('taravangian', "#e377c2"), ('lirin', "#bcbd22")] # szeth-taravangian the same color, lirin-kal the same color
color_dict = dict(names_to_colors)


# variables and method for saving plots
savePlots = False
folder = ''
plot_append = ''

def setPlotSaveVariables(save_plots, byPage):
    
    global savePlots
    global folder
    global plot_append
    
    savePlots = save_plots
    
    if byPage: 
        folder = './Images/ByPage/'
        plot_append = '_ByPage.png'
    else: 
        folder = './Images/ByChapter/'
        plot_append = '_ByChapter.png'
        
        
    if savePlots: print('Saving plots to: ', folder, ' with suffix: ', plot_append)
    else: print('NOT Saving plots to: ', folder, ' with suffix: ', plot_append)


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

    if len(df_with_results.index) < 500:
        plt.ylabel("Number of Chapters")
    else:
        plt.ylabel("Number of Pages")
           
    plt.xlabel("Character Cluster")

    if bookNum == 0: save_path = folder + 'kMeans_AllBooks' + plot_append
    else: save_path = folder + f'kMeans_Book{bookNum}' + plot_append
        
    if savePlots: 
        print('Saving image: ', save_path)
        plt.savefig(save_path)  
    
    plt.show()
    
    
# plot number of counts in each topic (max value) against booknum
def getNMFCounts(df_with_results, top_words, bookNum):

    unique_topic_values = df_with_results['NMF_Top_Topic'].sort_values().unique()
    labels = [top_words[item] for item in unique_topic_values]

    fig, ax = plt.subplots()        
    sns.countplot(x='NMF_Top_Topic', data=df_with_results.sort_values(by='NMF_Top_Topic'), palette=getPalette(labels)) # sort NMF top topic column by value in order to plot names and colors consistently
        
    ax.set_xticklabels(labels)
    
    if bookNum == 0:
        plt.title("All Books NMF")
    else:
        plt.title(f"Book {bookNum} NMF")
    
    if len(df_with_results.index) < 500:
        plt.ylabel("Number of Chapters")
    else:
        plt.ylabel("Number of Pages")
        
    plt.xlabel("Character Topic")

    if bookNum == 0: save_path = folder + 'NMF_AllBooks' + plot_append
    else: save_path = folder + f'NMF_Book{bookNum}' + plot_append
        
    if savePlots: 
        print('Saving image: ', save_path)
        plt.savefig(save_path)     
    
    plt.show()
    
    
# plot number of counts in each topic (max value) against booknum, for subtopic NMF results
def getNMFCounts_SubTopic(df_with_results, top_words, bookNum):

    unique_topic_values = df_with_results['NMF_SubTopic_Top_Topic'].sort_values().unique()
    labels = [top_words[item] for item in unique_topic_values]

    fig, ax = plt.subplots()        
    sns.countplot(x='NMF_SubTopic_Top_Topic', data=df_with_results.sort_values(by='NMF_SubTopic_Top_Topic'))#, palette=getPalette(labels)) # sort NMF top topic column by value in order to plot names and colors consistently
        
    ax.set_xticklabels(labels)
    
    if bookNum == 0:
        plt.title("All Books NMF")
    else:
        plt.title(f"Book {bookNum} NMF")
    
    if len(df_with_results.index) < 500:
        plt.ylabel("Number of Chapters")
    else:
        plt.ylabel("Number of Pages")
        
    plt.xlabel("Sub Topic")

    if bookNum == 0: save_path = folder + 'NMF_AllBooks' + plot_append
    else: save_path = folder + f'NMF_Book{bookNum}' + plot_append
        
    if savePlots: 
        print('Saving image: ', save_path)
        plt.savefig(save_path)     
    
    plt.show()    
    

# make MDS plot using vectorized matrix directly
def makeMDSPlot(vectorized_matrix, df_with_results, top_words):

    distances = cosine_distances(vectorized_matrix)
    
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=232, max_iter=300, verbose=1)
    positions_2d = mds.fit_transform(distances)
#     print('Final stress value: %f' %mds.stress_)

    df_temp = pd.DataFrame(positions_2d, columns=['MDS Axis 1', 'MDS Axis 2'])
    df_temp['Cluster'] = df_with_results['kMeans']
    df_temp['Label'] = df_temp['Cluster'].apply(lambda row: top_words[row])    
    
    labels = [top_words[item] for item in list(df_temp['Cluster'].sort_values().unique())]
    
    sns.lmplot(x='MDS Axis 1', y='MDS Axis 2', data=df_temp.sort_values(by='Cluster'), hue='Label', palette=getPalette(labels), fit_reg=False)

    plt.title("MDS All Books kMeans")
    
    save_path = folder + 'kMeans_MDS' + plot_append    
    if savePlots: 
        print('Saving image: ', save_path)
        plt.savefig(save_path, bbox_inches='tight')  
    
    plt.show()
    

# make tSNE plot using vectorized matrix directly for kMeans results
def makeTSNEPlot(vectorized_matrix, df_with_results, top_words, inputPerplexity=50):
    
#     distances = cosine_distances(vectorized_matrix)

    tsne = TSNE(n_components=2, verbose=1, perplexity=inputPerplexity, n_iter=1000, learning_rate=200, random_state=6321)
    tsne_results = tsne.fit_transform(vectorized_matrix)
#     tsne_results = tsne.fit_transform(distances)
    
    df_tsne = pd.DataFrame(tsne_results, columns=['tSNE Axis 1', 'tSNE Axis 2'])
    df_tsne['Cluster'] = df_with_results['kMeans']
    df_tsne['Label'] = df_tsne['Cluster'].apply(lambda row: top_words[row])

    labels = [top_words[item] for item in list(df_tsne['Cluster'].sort_values().unique())]
    
    sns.lmplot(x='tSNE Axis 1', y='tSNE Axis 2', data=df_tsne.sort_values(by='Cluster'), hue='Label', palette=getPalette(labels), fit_reg=False)

    plt.title("tSNE All Books kMeans")
    
    save_path = folder + 'kMeans_tSNE' + plot_append    
    if savePlots: 
        print('Saving image: ', save_path)
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    

# make tSNE plot using nmf topic vectors for each document, and using cosine distances in order to calculate the similarities
def makeTSNEPlotFromNMF(vectorized_matrix, df_with_results, top_words, inputPerplexity=50):

    nmf_topic_vectors_2d = np.stack(df_with_results['NMF'].to_numpy(), axis=0)
    distances = cosine_distances(nmf_topic_vectors_2d)

    tsne = TSNE(n_components=2, verbose=1, perplexity=inputPerplexity, n_iter=1000, learning_rate=200, random_state=6321)
    tsne_results = tsne.fit_transform(distances)
    
    df_tsne = pd.DataFrame(tsne_results, columns=['tSNE Axis 1', 'tSNE Axis 2'])
#     df_tsne['Topic'] = pd.Series([np.asarray(values).argmax() for values in df_with_results['NMF']]) # get best topic    
    df_tsne['Topic'] = df_with_results['NMF_Top_Topic'] # get best topic    
    df_tsne['Label'] = df_tsne['Topic'].apply(lambda row: top_words[row]) # find label for best topic

    labels = [top_words[item] for item in list(df_tsne['Topic'].sort_values().unique())]
    
    sns.lmplot(x='tSNE Axis 1', y='tSNE Axis 2', data=df_tsne.sort_values(by='Topic'), hue='Label', palette=getPalette(labels), fit_reg=False)

    plt.title("tSNE All Books NMF")
    
    save_path = folder + 'NMF_tSNE' + plot_append    
    if savePlots: 
        print('Saving image: ', save_path)
        plt.savefig(save_path, bbox_inches='tight')      
        
    plt.show()
    
    
# make word cloud plot from NMF results
def makeWordCloudPlot(word_dict, character, subTopicNum):
    
    wc = WordCloud(background_color="white", max_words=100)#, mask=alice_mask)
    
    wc.generate_from_frequencies(word_dict)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    
    save_path = folder + 'NMF_WordCloud_' + character + '_' + str(subTopicNum) + plot_append    
    if savePlots: 
        print('Saving image: ', save_path)
        plt.savefig(save_path)  
    
    plt.show()
    
    
    
def makeSubtopicTrendPlots(subtopic_list_counts_filled, character, top_words):
    
    for j, counts_data_list in enumerate(subtopic_list_counts_filled):
        x_val = [x[0] for x in counts_data_list]
        y_val = [x[1] for x in counts_data_list] 
        
        sns.lineplot(x=x_val, y=y_val)
        
        plt.title(f'{character}: {top_words[j]}')
        plt.xlabel('Global Chapter Number')
        plt.ylabel('Number of Pages')
        
        save_path = folder + character + 'SubtopicTrend_' + str(j) + plot_append
        if savePlots: 
            print('Saving image: ', save_path)
            plt.savefig(save_path, bbox_inches='tight')  
        
        plt.show()
        
        
def makeSubtopicJourneyPlot(subtopic_list_counts_filled, character, top_words):
        
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    
    for j, counts_data_list in enumerate(subtopic_list_counts_filled):
        x_val = [x[0] for x in counts_data_list]
        y_val = [x[1] for x in counts_data_list] 
        
        subtopic_word = top_words[j]
        
        sns.lineplot(x=x_val, y=y_val, label=subtopic_word)

        # attempts with stacking the lines
#         if j == 0:
#             ax1.fill_between(x_val, y_val, 0, alpha=1, label=subtopic_word)
#         else:
#             previous_y_val = [x[1] for x in subtopic_list_counts_filled[j-1]] 
#             ax1.fill_between(x_val, y_val, previous_y_val, alpha=1, label=subtopic_word)

    plt.legend()
    
    plt.title(f'{character}: Subtopic Journey')
    plt.xlabel('Global Chapter Number')
    plt.ylabel('Number of Pages')

    save_path = folder + character + 'Journey' + plot_append
    if savePlots: 
        print('Saving image: ', save_path)
        plt.savefig(save_path, bbox_inches='tight')        

    plt.show()
    
    
    
def sentimentPlot(inX, inY, character):
    
    sns.lineplot(x=inX, y=inY)
    
    plt.title(f'{character}: Sentiment Journey')
    plt.xlabel('Global Page Number')
    plt.ylabel('Sentiment Score')

    save_path = folder + character + 'SentimentJourney' + plot_append
    if savePlots: 
        print('Saving image: ', save_path)
        plt.savefig(save_path, bbox_inches='tight')        

    plt.show()
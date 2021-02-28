# Natural Language Processing on "The Stormlight Archive"

This project was developed over two and a half weeks as part of the [Metis](https://www.thisismetis.com/) data science boot-camp in Winter 2021.

### Objective:
---

Develop skills in NLP (natural language processing) while exploring a favorite fantasy series of mine. [The Stormlight Archive](https://www.brandonsanderson.com/the-stormlight-archive-series/) is an on-going high fantasy series by author Brandon Sanderson. Consisting of four books with over 400 chapters and 5000 pages, there was enough text for me to explore around the various main characters.

*Disclaimer: Nothing here is a major spoiler, but in case you are interested in reading these books, it might be best to look through this project after having done so!*



### Methodology and Results:
---

First, the four books of the series as purchased from Amazon were converted into pdf format and then read in to a jupyter notebook using [pdfplumber](https://github.com/jsvine/pdfplumber). The text was then processed in chapter or page-level increments, using nltk and standard text-preprocessing steps, ie. stemming, stop-word removal, punctuation removal, tokenization, etc. 

After the text was processed, I did kMeans clustering, and NMF and LDA topic modeling on the text. The NMF topic modeling performed the best without much tuning so I chose that as my main technique to run though the text with. (The kMeans clustering and LDA topic modeling performed okay, but the results were not as clean.)

 I found that the topic modeling naturally (and probably unsurprisingly) converged on the main characters themselves as the topics. The distribution of documents (chapters) as labeled by their top topic can be seen in the figure below, where I tuned the number of topics to 7 to align with the number of main characters, and I labeled each topic by the respetive main characters:

![](Images/ByPage/NMF_AllBooks_ByPage.png)


After the initial topic modeling, I did dimensionality reduction with MDS and tSNE. The tSNE plot for the page-level documents is seen below, using the NMF topic vectors per document with distances calculated as the cosine distances between documents. As can be seen, their are character clusters around the outside of the plot, with some interaction between the various blobs. At the middle there is a region where the NMF topic modeling had more difficulty selecting the particular topic or character, and these points can be understood as those pages where the characters in the books were together for whatever plot point was then occurring.

![](Images/ByPage/NMF_tSNE_ByPage.png)


After exploring some dimensionality reduction, I decided to perform further topic modeling on the main characters themselves after I had added names to the stop-words list, in order to select on some more thematic topics. The plot below shows a word cloud for the character "Shallan," for one of her main topics. Shallan is an artist in the books, and the topic modeling found this pretty readily.

![](Images/ByPage/NMF_WordCloud_Shallan_4_ByPage.png)

This last plot then shows the trend of Shallan's journey throughout the book in terms of her sub-topics. Clusters of pink at the beginning, blue, orange, and green in the middle, and then red at the end correspond to various plot points and character moments For instance the pink in the beginning labeled by "book" references to her leisure times at the beginning of the story when she's just starting out, and the red "radiant" parts at the end refer to her most-recent, intense journey to grow and become a hero.

![](Images/ByPage/ShallanJourney_ByPage.png)


As a last step, I did attempt some sentiment analysis of the chapters and pages for each character using TextBlob and vaderSentiment. While the graphs came out okay-looking, when cross-referenced to the text the trends didn't make much sense, and I ultimately decided to scrap that part of the project. With more time and perhaps a finer document level (paragraph or perhaps sentences), this might be worth exploring more.




### Tools and Techniques:
---

- Text preprocessing: [pdfplumber](https://github.com/jsvine/pdfplumber), nltk (stemming, tokenization, stop-word removal, etc.)
- Text vectorization with CountVectorizer and TfidfVectorizer
- kMeans clustering
- NMF and LDA topic modeling
- Word embedding via MDS and tSNE
- Visualization via Matplotlib and Seaborn
- Sentiment analysis via TextBlob and vaderSentiment



### File Details:
---

- `code/`

	- `doNLP.ipynb` - main code which does various clustering, topic modeling, dimensionality reduction, and sentiment analysis on the four books 
	- `nlpUtils.py` - file which provides the methods for the clustering and topic modeling routines
	- `plotUtils.py` - file which provides various plotting methods, including word embedding plots, word clouds, etc.
	- `extractAndCleanText.ipynb` - notebook for extracting and cleaning the book text in chapter level increments
	- `extractAndCleanTextByPage.ipynb` - notebook for extracting and cleaning the book text in page level increments
	- `extractText.ipynb` - notebook for extracting and the book text in chapter level increments, with no text preprocessing
	- `nlpEDA.ipynb` - notebook used for initial NLP EDA and parameter optimization 
	- `sentimentEDA.ipynb` - notebook used for initial sentiment analysis EDA (more was eventually done at the bottom of the main `doNLP.ipynb` file)
	- `bookChaptersPageNumbers.py` - file which provided the page numbers for chapters, used when reading in and processing the book text

	- `sla_chapter_text.pkl` - pickled pandas dataframe containing processed chapter-level text, one chapter per row
	- `sla_page_text.pkl` - pickled pandas dataframe containing processed page-level text, one page per row
	- `sla_chapter_text_unprocessed.pkl` - pickled pandas dataframe containing unprocessed chapter-level text, one chapter per row



- `StormlightArchiveBooks/`

	- This folder contains the Stormlight Archive books (1-4) by Brandon Sanderson in PDF format on my own computer. I purchased these books through Amazon and used them in this NLP project. For rights reasons I am not committing the books themselves to this public repository.

	- `BookPageInfo.xlsx` - excel file containing information related to the book chapter and relevant pages, along with a "global chapter number" variable that was useful in cross-referencing the text
	

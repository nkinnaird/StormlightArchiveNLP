# Natural Language Processing on "The Stormlight Archive"

readme is a work in progress

This project was developed over two and a half weeks as part of the [Metis](https://www.thisismetis.com/) data science boot-camp in Winter 2021.

### Objective:
---


[The Stormlight Archive](https://www.brandonsanderson.com/the-stormlight-archive-series/)



### Methodology and Results:
---




### Tools and Techniques:
---

- text preprocessing: [pdfplumber](https://github.com/jsvine/pdfplumber), stemming, stop word removal, text cleaning, etc.
- text vectorization via CountVectorizer and TfidfVectorizer
- kMeans clustering
- NMF and LDA topic modeling and dimensionality reduction
- Word embedding via MDS and tSNE
- Visualization via matplotlib and seaborn
- Sentiment analysis via TextBlob and VaderSentiment






### File Details:
---

- `code/`

	- `doNLP.ipynb` - main code which fits the classification models on the train and test data, outputs plots and results
	- `nlpUtils.py` - file which provides various plotting and metric evaluation methods
	- `plotUtils.py` - file written by Kyle Gilde which provides feature names after data has been passed through `ColumnTransformer` pre-processors (including One-Hot Encoding steps)
	- `extractAndCleanText.ipynb` - notebook for
	- `extractAndCleanTextByPage.ipynb` - notebook for
	- `extractText.ipynb` - notebook for
	- `nlpEDA.ipynb` - notebook used for 
	- `sentimentEDA.ipynb` - notebook used for 
	- `bookChaptersPageNumbers.py` - file which provides various plotting and metric evaluation methods

	- `sla_chapter_text.pkl` - pickled df..
	- `sla_page_text.pkl` - pickled df..
	- `sla_chapter_text_unprocessed.pkl` - pickled df..



- `StormlightArchiveBooks/`

	- In my own repository, contains the four books..., see the readme here for more details

	- `BookPageInfo.xlsx` - contains info I generated...
	

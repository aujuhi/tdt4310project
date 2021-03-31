# Intelligent Text Analytics and Language Understanding (TDT4310) - Final Project
## Lexicon based analysis of topic related word usage 

Anke Unger
ankeju@stud.ntnu.no

### About
This repository contains tools to analyse the word usage of articles related to the corona crisis and the
LGBT-community  
The main goal was to create analysis tools for a contentbased examination of newsarticels.
This allows an easy inspection of the development over time and gives an overview over the sentiment 
and opinion in a large cluster of publications. As the approach is supposed to
work on varying sets of newsarticles, the tools are kept adaptable and are being evaluated with different
sorts of data. The focus lays on analysing the usage of vocabulary, which could be classified as offensive
or biased and to give a document based overview on the sentiment of articles. Recent methodologies for bias or 
aggressiveness detection consist of supervised learning techniques. Supervised learning is dependent on the existence
of labeled data to train the classifiers but most datasets are only for selected use cases, for example the
classification of tweets, political speeches or movie reviews. If the training data is not representative for
the real world data the algorithm will create poor results. This is especially the case for highly domain
specific texts. For incident a financial article will use a different vocabulary than a boulevard magazine
or a politician during a speech. This work therefore applies word lexicons, to analyse the frequency
of usage and interpreting the measured frequencies as hints for offensive, biased, sentimental or simply
controversial topics. A lexicon is easier and faster to create and maintained than a labeled dataset and
the quality of it can be easily controlled. By that the approach is far more flexible for different use cases,
especially when the lexicon is automatically created or adapted, as it can be “on the fly” build for every
different domain.

### Prerequisites
*`Python 3.0` or further  
*`anaconda` as a suggestion  
*further requirements are stated in the .requirement files

### Getting Started

#### Installation

pip install --user --requirement requirements.txt

#### Data
the crawled articles need to be placed in the /data folder. You can choose between different formats but the default setting is a reader function for .json files for the articles regarding corona and a .csv-file reader for articles regarding lgbt. I suggest using [news please](https://github.com/fhamborg/news-please) as it produces crawled articles directly in the right format. A couple of example articles to test the code are already in the data-folder.


#### Lexicon
To analyze the word usage with a different focus add a lexicon in the /lexica folder. Offensivenes, bias and sentiment are already placed there.

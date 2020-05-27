# Intelligent Text Analytics and Language Understanding (TDT4310) - Final Project
## Lexicon based analysis of topic related word usage 

Anke Unger
ankeju@stud.ntnu.no

### About
This repository contains tools to analyse the word usage of topic related articles in comparison
to non-topic related publications on data related to the corona crisis and the
LGBT-community. 

### Prerequisites
*`Python 3.0` or further  
*The usage of `anaconda` is suggested
for further requirements, please read the requirement files in the corresponding directory

### Getting Started

#### Data
the crawled articles need to be placed in the data/ folder. You can choose between different formats but the default setting is a reader function for json formats 
for the articles regarding corona and a reader for csv files for articles regarding lgbt. I suggest using [news please](https://github.com/fhamborg/news-please) as it produces 
crawled articles directly in the right format. A couple of example articles to test the code are already in the data-folder


#### Lexicon
To analyze the word usage with a different focus add a lexicon in the /lexica folder. Offensivenes, bias and sentiment are already placed there.

# Intelligent Text Analytics and Language Understanding (TDT4310) - Final Project
## Lexicon based analysis of topic related word usage 

Anke Unger
ankeju@stud.ntnu.no

### About
This repository contains tools to analyse the word usage of articles related to the corona crisis and the
LGBT-community 

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

from wordcloud import WordCloud
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from nltk.corpus import stopwords
import plotsAndDataInterpretation

"""
data folder must contain seedlists (unless hard coded), the lexicons and newsarticle files
"""
data_folder = "data/" # must contain the scraped articles as json txt or csv file. Just adapt the reader
                                     # function accordingly
lex_folder = "lexica/" # must contain the sentiment, bias and offensiveness lexica, as well as the seed-
                                      # lists of topics of your choice
result_folder = "results/" #the wordclouds, csv results and lineplots will appear here
alphabet_regex = re.compile('[^a-zA-Z0-9 -]')


def analyze_context(articles, seed_words, _filename):
    """
    Analyses the context of words in seedlist.
    :param articles: the list of all topicrelated articles
    :param seed_words: The words of which the context should be analysed
    :param _filename: Name of the resulting png
    :return: None
    saves wordclouds as png and prints most common contextwords with frequency
    """
    context_words = []
    # filtering of high frequent but uninformative words/ stop words
    uselessWords = set(stopwords.words('english'))
    # analyze context words for each epoch
    for ws in articles:
        for w in ws:
            if w in uselessWords:
                ws.remove(w)
        for w in ws:
            if w in uselessWords:
                ws.remove(w)
        for i, w in enumerate(ws):
            if w in seed_words:
                start = max(0, i - 3)
                end = min(i + 4, len(ws))
                context_words += ws[start:i]
                context_words += ws[i + 1:end]
    for w in context_words:
        if w in uselessWords:
            context_words.remove(w)
    # create wordcloud
    wc = WordCloud(collocations=False).generate(str(context_words))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(result_folder + _filename + '_wcContext.png')
    print('Number of context words', len(context_words))
    print('Unique context words', len(set(context_words)))
    counts = Counter([i for i in context_words])
    print('Most 10 common context words', counts.most_common(10))


def analyze_sentiment(articles, _filename):
    """
    Analyses the frequency of occurence of sentiment words
    :param articles: list or bucket of articles
    :param _filename: name of the article file and custom sentiment
    :return: number of positive and negative sentiment words
    """
    if not ".csv" in _filename:
        _filename = _filename + ".csv"
    if os.path.exists(lex_folder + 'sentiment_lexicon_' + _filename):
        print("Using custom Lexicon...")
        sentiment_words = pd.read_csv(lex_folder + 'sentiment_lexicon_' + _filename, header=0)
        pos_sentiment = dict(zip([w.lower() for w in sentiment_words["positive"].fillna('').tolist() if w != '']
                                 , [w for w in sentiment_words["pos_iwf"].fillna('').tolist() if w != '']))
        neg_sentiment = dict(zip([w.lower() for w in sentiment_words["negative"].fillna('').tolist() if w != '']
                                 , [w for w in sentiment_words["neg_iwf"].fillna('').tolist() if w != '']))
    else:
        print("Using default lexicon...")
        sentiment_words = pd.read_csv(lex_folder + 'sentiment_lexicon.csv', skiprows=[1], header=0)
        pos_sentiment = dict(zip(
            [w.lower() for w in sentiment_words["Positive sentiment"].fillna('').tolist() if w != ''],
            np.ones(len(sentiment_words["Positive sentiment"]))))
        neg_sentiment = dict(zip(
            [w.lower() for w in sentiment_words["Negative sentiment"].fillna('').tolist() if w != ''],
            np.ones(len(sentiment_words["Negative sentiment"]))))
    countsPL, countsNL = find_words_with_negations(articles, pos_sentiment, neg_sentiment)
    no_pos = sum([pos_sentiment[v] for v in countsPL.keys()])
    no_neg = sum([neg_sentiment[v.lower()] for v in countsNL.keys()])
    print('10 most common positive sentiment words: ', countsPL.most_common(10))
    print('10 most common negative sentiment words: ', countsNL.most_common(10))
    print('Score positive words: ', no_pos)
    print('Score negative words: ', no_neg)
    return no_pos, no_neg


def analyze_bias(articles, biased_words):
    """
    Analyses the frequency of occurence of biased words
    :param articles: list or bucket of articles
    :param biased_words: list of biased words
    :return: the number of biased words in articles
    """
    biased = find_words(articles, biased_words)
    no_biased = sum(biased.values())
    print('Most common biased words in articles: ', biased.most_common(10))
    print('Number biased words in articles ', no_biased)
    return no_biased


def analyze_offensiveness(articles, offensive_words):
    """
    Analyses the frequency of occurence of offensive words
    :param articles: list or bucket of articles
    :param biased_words: list of offensive words
    :return: the number of offensive words in articles
    """
    offensive = find_words(articles, offensive_words)
    no_offensive = sum(offensive.values())
    print('Most common offensive words: ', offensive.most_common(10))
    print('Number offensive words in articles ', no_offensive)
    return no_offensive


def find_words(articles, lexicon):
    """
    searches for words in lexicon
    :param articles: list or bucket of articles
    :param lexicon: the lexicon with words to find
    :return: counter object
    """
    lst = []
    for i, x in enumerate(articles):
        for ns in lexicon:
            ns = ns.lower()
            if ns in x:
                lst.append(ns)
    return Counter([i for i in lst])


def find_words_with_negations(articles, lexicon_pos, lexicon_neg):
    """
    searches for words in lexicon, ignores negated words
    :param articles: list or bucket of articles
    :param lexicon_pos: sentiment lexicon with positive words
    :param lexicon_neg: sentiment lexicon with negative words
    :return: counter object
    """
    lst_pos = []
    lst_neg = []
    for x in articles:
        for ns in lexicon_pos.keys():
            ns = ns.lower()
            if ns in x:
                position = x.index(ns)
                if position > 0:
                    if any(negation in x[position - 1] for negation in ["not", "n't", "no"]):
                        continue
                lst_pos.append(ns)
        for ns in lexicon_neg.keys():
            ns = ns.lower()
            if ns in x:
                position = x.index(ns)
                if position > 0:
                    if any(negation in x[position - 1] for negation in ["not", "n't", "no"]):
                        continue
                lst_neg.append(ns)
    return Counter([i for i in lst_pos]), Counter([i for i in lst_neg])


def get_text_txt(file_path, seed_list):
    """
    loads articles from txt files
    :param file_path: name of article files
    :param seed_list: topic related seed list
    :return: list of article texts and publish dates, split by topic
    """
    f = open(data_folder + file_path, "r")
    text = f.read().split("|")
    date = ""
    datelist_bkg = []
    artlist_bkg = []
    datelist_cov = []
    artlist_cov = []
    for i, line in enumerate(text):
        if (i % 2) == 0:
            date = line
        else:
            if any(x in line for x in seed_list):
                artlist_cov.append(alphabet_regex.sub('', line).lower().strip().split())
                datelist_cov.append(date)
            else:
                if line:
                    artlist_bkg.append(alphabet_regex.sub('', line).lower().strip().split())
                    datelist_bkg.append(date)
    return artlist_bkg, datelist_bkg, artlist_cov, datelist_cov


def get_text_json(file_path, seed_list):
    """
    loads articles from json files
    :param file_path: name of article files
    :param seed_list: topic related seed list
    :return: list of article texts and publish dates, split by topic
    """
    datelist_bkg = []
    artlist_bkg = []
    datelist_cov = []
    artlist_cov = []
    for j_name in os.listdir(data_folder + file_path):
        if ".json" in j_name:
            filename = data_folder + file_path + j_name
            with open(filename) as json_file:
                try:
                    data = json.load(json_file)
                except json.decoder.JSONDecodeError:
                    continue
                text = data["maintext"]
                date = data["date_publish"].split(" ")[0]
                if text and type(text) == str and len(text) > 0:
                    if any(x in text for x in seed_list):
                        artlist_cov.append(alphabet_regex.sub('', text).lower().strip().split())
                        datelist_cov.append(date)
                    else:
                        artlist_bkg.append(alphabet_regex.sub('', text).lower().strip().split())
                        datelist_bkg.append(date)
                else:
                    continue
    return artlist_bkg, datelist_bkg, artlist_cov, datelist_cov


def count_words(artlist):
    """
    counts the words in list of articles
    :param artlist: list of articles
    :return: number of words
    """
    i1 = 0
    for k in artlist:
        for w in k:
            i1 += 1
    print("No of words in article: ", i1)
    return i1


def create_buckets(articles, dates, stepSize="Day"):
    """
    splits data in to buckets
    :param articles: list of articles
    :param dates: publishing date in same order
    :param stepSize: optional bucket size
    :return: list of buckets of articles
    """
    artPerDay = {}
    if stepSize.lower() == "month":
        dates = [date.split("-")[1] for date in dates]
    if stepSize.lower() == "year":
        dates = [date.split("-")[0] for date in dates]
        print(dates)
    for i, date in enumerate(dates):
        if date in artPerDay:
            a = artPerDay[date]
            a.append(articles[i])
        else:
            artPerDay[date] = [articles[i]]
    return artPerDay


def analyseArticlesCorona():
    """
    controller function to call all necessary steps to analyse articles about corona
    :return: None
    """
    covid_seeds = ["covid", "cov", "corona", "sars", "outbreak", "pandemic", "virus"]
    articles_fox_path = "coronaExample/"  # json file format

    fox_artlist_bkg, fox_datelist_bkg, fox_artlist_cov, fox_datelist_cov = get_text_json(
        articles_fox_path, covid_seeds)

    analyze_context(fox_artlist_cov, covid_seeds, "resultFoxCov")

    fox_grnd_perday = create_buckets(fox_artlist_bkg, fox_datelist_bkg)
    fox_cov_perday = create_buckets(fox_artlist_cov, fox_datelist_cov)
    art_list = [fox_artlist_bkg, fox_artlist_cov]
    resultFiles = ["resultFoxBkg", "resultFoxCov"]
    dicts = [fox_grnd_perday, fox_cov_perday]
    for i in range(len(dicts)):
        plotsAndDataInterpretation.learn_domain_sentiment_words(art_list[i], resultFiles[i])
        plotsAndDataInterpretation.create_result_file(resultFiles[i], dicts[i])
        plotsAndDataInterpretation.plotFreqResults(resultFiles[i])
    print("Resultfiles and plots have been created in " + result_folder)


def analyseArticlesLGBTQ():
    """
    controller function to call all necessary steps to analyse articles about corona
    :return: None
    """
    lgbtq = pd.read_csv(data_folder + 'lgbtExample/lgbt_news_corpus.csv', encoding="ISO-8859-1")
    lgbtq_articles_and_date = [(alphabet_regex.sub('', row.Text).lower().strip().split(), row[5])
                               for row in lgbtq.itertuples()
                               if type(row.Text) == str
                               and len(row.Text) > 0
                               and '1986-00-00T00:00:00Z' < row[5] <= '2017-00-00T00:00:00Z']
    article_list, date_list = zip(*lgbtq_articles_and_date)
    backg = pd.read_csv(data_folder + 'lgbtExample/background_news_corpus.csv', encoding="ISO-8859-1")
    backg_articles_and_date = [(alphabet_regex.sub('', row.Text).lower().strip().split(), row[5])
                               for row in backg.itertuples()
                               if type(row.Text) == str
                               and '1986-00-00T00:00:00Z' < row[5] <= '2017-00-00T00:00:00Z']
    backg_article_list, backg_date_list = zip(*backg_articles_and_date)
    with open(lex_folder + 'seed_list.txt', 'r') as fp:
        seed_words = [i for i in fp.read().split('\n') if len(i) > 0]
    analyze_context(article_list, seed_words, "lgbtq_new")
    lgbtq_articles_per_year = create_buckets(article_list, date_list, stepSize="year")
    bkg_articles_per_year = create_buckets(backg_article_list, backg_date_list, stepSize="year")
    resultFiles = ["resultLGBTQBkg", "resultLGBTQ"]
    art_list = [backg_article_list, article_list]
    dicts = [bkg_articles_per_year, lgbtq_articles_per_year]
    for i in range(len(dicts)):
        plotsAndDataInterpretation.learn_domain_sentiment_words(art_list[i], resultFiles[i])
        plotsAndDataInterpretation.create_result_file(resultFiles[i], dicts[i])
        plotsAndDataInterpretation.plotFreqResults(resultFiles[i])
        plotsAndDataInterpretation.plotSentResults(resultFiles[i])
    print("Resultfiles and plots have been created in " + result_folder)


if __name__ == '__main__':
    analyseArticlesCorona()
    analyseArticlesLGBTQ()

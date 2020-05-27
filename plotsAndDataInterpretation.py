from matplotlib import pyplot as plt
import numpy as np
import articleAnalysis
import pandas as pd
import csv
import seaborn as sns
import nltk
import re
from statsmodels.tsa.seasonal import seasonal_decompose
sns.set(style="darkgrid")


result_folder = articleAnalysis.result_folder


def plotFreqResults(csvPath):
    """
    plots bias and offensiveness results from csv result files as png
    :param csvPath: path to csv result files
    :return: None
    """
    if not ".csv" in csvPath:
        csvPath = csvPath + ".csv"
    _data = pd.read_csv(result_folder + csvPath)
    vars = ["bias"] * len(_data["bias"]) + ["offensive"] * len(_data["offens"])
    day = list(_data["date"])
    day.extend(_data["date"])
    vals = list(_data["bias"])
    vals.extend(list(_data["offens"]))
    data_preproc = pd.DataFrame({
        'Day': day,
        'value': vals,
        'variable': vars
    })
    if "Fox" in csvPath:
        _ticks = np.arange(0, 120, 5)
    plt.figure(figsize=(12, 9))
    with sns.plotting_context("notebook", font_scale=2.0):
        ax = sns.lineplot(x='Day', y='value', hue='variable', data=data_preproc)
        if "Fox" in csvPath:
            plt.xticks(ticks=_ticks, fontsize=13, rotation=60)
        fig = ax.get_figure()
        ax.set(xlabel='Date', ylabel='Normalized frequency')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        result_name = csvPath.split(".")[0] + ".png"
        fig.savefig(result_folder + result_name)


def plotSentResults(csvPath):
    """
    plots positive and negative sentiment results from csv result files as png
    :param csvPath: path to csv result files
    :return: None
    """
    if not ".csv" in csvPath:
        csvPath = csvPath + ".csv"
    _data = pd.read_csv(result_folder + csvPath)
    vars = ["positive"] * len(_data["pos"]) + ["negative"] * len(_data["neg"])
    day = list(_data["date"])
    day.extend(_data["date"])
    vals = list(_data["pos"])
    vals.extend(list(_data["neg"]))
    data_preproc = pd.DataFrame({
        'Day': day,
        'value': vals,
        'variable': vars
    })
    if "Fox" in csvPath:
        _ticks = np.arange(0, 120, 5)
    plt.figure(figsize=(12, 9))
    with sns.plotting_context("notebook", font_scale=2.0):
        ax = sns.lineplot(x='Day', y='value', hue='variable', data=data_preproc)
        if "Fox" in csvPath:
            plt.xticks(ticks=_ticks, fontsize=13, rotation=60)
        fig = ax.get_figure()
        ax.set(xlabel='Date', ylabel='Weighted frequency')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        result_name = csvPath.split(".")[0] + "Sentiment" + ".png"
        fig.savefig(result_folder + result_name)


def plot_trend_analysis(csvPath, trgt_clmn):
    """
    Plots trend analysis for one column
    :param csvPath: Path to csv result file
    :param trgt_clmn: targeted column name in result file
    :return: None
    """
    if not ".csv" in csvPath:
        csvPath = csvPath + ".csv"
    df = pd.read_csv(result_folder + csvPath, parse_dates=['date'], index_col='date')
    result_add = seasonal_decompose(df[trgt_clmn].values, model='additive', period=15, extrapolate_trend='freq')
    plt.rcParams.update({'figure.figsize': (20, 20)})
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    result_name = csvPath.split(".")[0] + trgt_clmn + "_Trend" + ".png"
    plt.savefig(result_folder + result_name)


def collectFreqResults(dic_perday, _filename):
    """
    Calls different frequency analysis and collects results
    :param dic_perday: dictionary with date as key and articles as value
    :param _filename: name of created resultfiles
    :return: list of results of frequency analysis
    """
    biased_words = pd.read_csv(articleAnalysis.lex_folder + 'bias_lexicon.csv', skiprows=[1, 2, 3, 4, 5], header=0)
    biased_words = biased_words['LWIC negative emotion (negemo), swear, anger, sad, death'].tolist()
    with open(articleAnalysis.lex_folder + "offensive_lexicon.txt", "r") as fp:
        offensive_words = [i for i in fp.read().split('\n') if len(i) > 0]
    result = []
    for date, artlist in dic_perday.items():
        x = []
        x.append(date)
        no_words = articleAnalysis.count_words(artlist)
        pos_sen, neg_sen = articleAnalysis.analyze_sentiment(artlist, _filename)
        x.append(pos_sen / no_words)
        x.append(neg_sen / no_words)
        x.append(articleAnalysis.analyze_bias(artlist, offensive_words) / no_words)
        x.append(articleAnalysis.analyze_offensiveness(artlist, biased_words) / no_words)
        result.append(x)
    return result


def create_result_file(_filename, dic_perday):
    """
    Navigates the analysis for each bucket and writes result file as csv
    :param _filename: name of created result file
    :param dic_perday: dictionary with date as key and articles as value
    :return: None
    """
    if not (".csv") in _filename:
        _filename = _filename + ".csv"
    f = open(result_folder + _filename, "w+", newline='')
    file_writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    file_writer.writerow(["date", "pos", "neg", "bias", "offens"])
    lines = collectFreqResults(dic_perday, _filename)
    for l in lines:
        file_writer.writerow(l)
    f.close()


def learn_domain_sentiment_words(articles, _filename):
    """
    learns domain specific lexicon of sentiment words and saves as csv
    :param articles: articles to learn words from
    :param _filename: name of the result file
    :return: None
    """
    # Invers freq = log(Anzahl docs / Anzahl docs mit Term)
    sentiment_words = pd.read_csv(articleAnalysis.lex_folder + 'sentiment_lexicon.csv', skiprows=[1], header=0)
    pos_sentiment = [w for w in sentiment_words['Positive sentiment'].fillna('').tolist() if w != '']
    neg_sentiment = [w for w in sentiment_words['Negative sentiment'].fillna('').tolist() if w != '']
    doc_freq_pos = dict.fromkeys(pos_sentiment, 1)
    doc_freq_neg = dict.fromkeys(neg_sentiment, 1)
    for a in articles:
        new_pos, new_neg = find_conjunctions(a, pos_sentiment)
        new_neg_2, new_pos_2 = find_conjunctions(a, neg_sentiment)
        new_pos.update(new_pos_2)
        new_neg.update(new_neg_2)
        for w in new_pos:
            if w in doc_freq_pos:
                doc_freq_pos[w] += 1
            else:
                doc_freq_pos[w] = 1
        for w in new_neg:
            if w in doc_freq_neg:
                doc_freq_neg[w] += 1
            else:
                doc_freq_pos[w] = 1
    no_art = len(articles)
    print("Save dictionary at ", articleAnalysis.lex_folder)
    # write newly learned lexicon to file
    if not (".csv") in _filename:
        _filename = _filename + ".csv"
    f = open(articleAnalysis.lex_folder + "sentiment_lexicon_" + _filename, "w+", newline='', encoding="UTF-8"
             )
    file_writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    file_writer.writerow(["positive", "pos_iwf", "negative", "neg_iwf"])
    for i in range(max(len(doc_freq_neg), len(doc_freq_pos))):
        if i < min(len(doc_freq_neg), len(doc_freq_pos)):
            file_writer.writerow(
                [list(doc_freq_pos)[i],
                 np.log10(no_art / list(doc_freq_pos.values())[i]),
                 list(doc_freq_neg)[i],
                 np.log10(no_art / list(doc_freq_neg.values())[i])])
        else:
            if len(doc_freq_neg) == max(len(doc_freq_neg), len(doc_freq_pos)):
                file_writer.writerow(
                    ["",
                     "",
                     list(doc_freq_neg)[i],
                     np.log10(no_art / list(doc_freq_neg.values())[i])])
            else:
                file_writer.writerow(
                    [list(doc_freq_pos)[i],
                     np.log10(no_art / list(doc_freq_pos.values())[i]),
                     "",
                     ""])
    f.close()


def find_conjunctions(article, lexicon):
    """
    finds conjunctions in text and return conjuncted adjectives
    :param article: articles to analyse
    :param lexicon: predefined sentiment lexicon
    :return: Positive and negative extended list of sentiment words
    """
    new_sent_1 = set()
    new_sent_2 = set()
    tagged_article = nltk.pos_tag(article, tagset='universal')
    for ns in lexicon:
        ns = ns.lower()
        if ns in article:
            new_sent_1.add(ns)
            position = article.index(ns)
            if position > 2:
                if "and" in article[position - 1][0] \
                        and "ADJ" in tagged_article[position - 2][1] \
                        and len(tagged_article[position - 2][0]) > 1 \
                        and bool(re.search(r'\d', article[position - 2])):
                    new_sent_1.add(article[position - 2])
                if "but" in article[position - 1] \
                        and "ADJ" in tagged_article[position - 2][1] \
                        and len(tagged_article[position - 2][0]) > 1 \
                        and bool(re.search(r'\d', article[position - 2])):
                    new_sent_2.add(article[position - 2])
            if position < len(article) - 2:
                if "and" in article[position + 1] \
                        and "ADJ" in tagged_article[position + 2][1] \
                        and len(tagged_article[position + 2][0]) > 1 \
                        and bool(re.search(r'\d', article[position + 2])):
                    new_sent_1.add(article[position + 2])
                if "but" in article[position + 1] \
                        and "ADJ" in tagged_article[position + 2][1] \
                        and len(tagged_article[position + 2][0]) > 1 \
                        and bool(re.search(r'\d', article[position + 2])):
                    new_sent_2.add(article[position + 2])
    return new_sent_1, new_sent_2

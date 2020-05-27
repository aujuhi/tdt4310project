import newspaper
import datetime
import pytz

sources = ["cnn"]
#sources from 23.01.2020 to 11.05.2020
for source in sources:
    cnn_paper = newspaper.build("https://www." + source + ".com" , memoize_articles=False, language = 'en')

    text_file_after = open("C:/Users/Anke/PycharmProjects/tdt4310project/data/articles_after_"
                           + source + ".txt", "a+")
    text_file_before = open("C:/Users/Anke/PycharmProjects/tdt4310project/data/articles_before_"
                            + source + ".txt", "a+")
    for first_article in cnn_paper.articles:
        first_article.download()
        try:
            first_article.parse()
        except newspaper.article.ArticleException:
            print("Article `download()` failed with 403 Client Error: Forbidden for url: "
                  + first_article.url)
            continue
        pub_date = first_article.publish_date
        art_text = first_article.text
        if len(first_article.text) > 300 \
                and pub_date:
            if pub_date.replace(tzinfo=pytz.UTC) > datetime.datetime(2020, 1, 23).replace(tzinfo = pytz.UTC):
                try:
                    text_file_after.write(pub_date.strftime("%m/%d/%Y"))
                    text_file_after.write(first_article.text)
                except UnicodeEncodeError:
                    print(first_article.text)
            if first_article.publish_date.replace(tzinfo=pytz.UTC) < datetime.datetime(2019, 11, 30).replace(tzinfo = pytz.UTC):
                try:
                    text_file_before.write(first_article.text + "|")
                except UnicodeEncodeError:
                    print(first_article.text)
    text_file_after.close()
    text_file_before.close()
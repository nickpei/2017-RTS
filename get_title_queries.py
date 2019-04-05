import json
import nltk
import string
import time
import tweepy
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def map_stems(text):
    map = {}
    tokens = nltk.word_tokenize(text)
    for item in tokens:
        map[stemmer.stem(item)] = item
    return map


def get_titles_dict(orignal_query_file):
    titles_dict = {}

    with open(orignal_query_file) as f:
        profiles = json.load(f)
        for prof in profiles:
            titles_dict[prof["topid"]] = prof["title"]
    return titles_dict


def get_title_queries(titles_dict):
    title_queries = {}
    title_tweets = {}
    start_time = time.time()

    api_key = 'Wxlc55Y4Q72WRzuwwgb2p6eue'
    api_secret = 'rcnNq1cwCHZG4VVBa0uUv0gX3fhGg3M0R5bIVuIa6kYibnVvzQ'
    access_token = '850839890296270857-CNCMC6vdZys7cC9lfWGOkQoJPdbg5mu'
    access_token_secret = 'NO8rO5OGSyitVoscPsGDnwMmZXZ4FBuGPG56HwvjjtinT'

    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    # limit_status = api.rate_limit_status()
    # print(limit_status)
    # print('line 127')

    for key, value in titles_dict.items():

        query = tuple(value.translate(str.maketrans('', '', string.punctuation)).split())
        max_tweets = 100
        searched_tweets = []

        search_result = api.search(q=query, count=max_tweets)
        for tweet in search_result:
            if (not tweet.retweeted) and ('RT @' not in tweet.text) and (tweet.lang == "en"):
                searched_tweets.append(tweet.text)

        if searched_tweets:  # check if tweets found
            text = re.sub(r"http\S+", "", str(searched_tweets))  # remove urls

            title_tweets[key] = text

        # calculate tf-idf score for each term in all searched tweets
            special_char = ["‘","’","·","–","“","”"]
            no_punctuation = str(text).translate(str.maketrans('', '', string.punctuation)).translate({ord(c): 'special char' for c in special_char}).lower()

            mapped_stems = map_stems(no_punctuation)

            tfidf_dict = {}
            tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
            tfs = tfidf.fit_transform([no_punctuation])
            feature_names = tfidf.get_feature_names()
            for col in tfs.nonzero()[1]:
                tfidf_dict[feature_names[col]] = tfs[0, col]

            # get top 10 title terms as expanded query

            expanded_query = ''
            sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
            top_10_elements = sorted_tfidf[:10]
            for item in top_10_elements:
                expanded_query = mapped_stems[item[0]] + ' ' + expanded_query

            title_queries[key] = expanded_query

            time.sleep(5.05)

        else:
            # title_queries[key] = 'No query from tweets'
            title_queries[key] = ''
            time.sleep(5.05)

        elapsed_time = time.time() - start_time
        print(elapsed_time)
    with open('title_tweets.json', 'w', encoding='utf-8') as f:
        json.dump(title_tweets, f, ensure_ascii=False, indent=1)

    return title_queries


def get_title_queries_json(title_queries):

    title_queries_json = []
    for key, value in title_queries.items():
        single = {"topid": key, "title": value}
        title_queries_json.append(single)

    return title_queries_json


orignal_query_file = "judged_topics"


def main():
    titles_dict = get_titles_dict(orignal_query_file)

    title_queries = get_title_queries(titles_dict)

    title_queries_json = get_title_queries_json(title_queries)

    with open('title.json', 'w', encoding='utf-8') as f:
        json.dump(title_queries_json, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
        main()

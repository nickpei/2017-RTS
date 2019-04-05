import json
import nltk
import string
import time
import requests
from bs4 import BeautifulSoup
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from fake_useragent import UserAgent
from collections import defaultdict

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


def get_narr_dict(orignal_query_file):
    narr_dict = {}

    with open(orignal_query_file) as f:
        profiles = json.load(f)
        for prof in profiles:
            narr_dict[prof["topid"]] = prof["narrative"]
    return narr_dict


def get_title_list(orignal_query_file):
    title_list = []
    with open(orignal_query_file) as f:
        profiles = json.load(f)
        for prof in profiles:
            title_list.append(prof["title"])

    return title_list


def get_narr_queries(narr_dict):
    narr_queries = {}
    narr_snippets = defaultdict(dict)
    title_num = 0
    # ua = UserAgent(cache=False)
    ua = UserAgent(cache=True)
    start_time = time.time()

    title_list = get_title_list(orignal_query_file)

    for key, value in narr_dict.items():

        print("Topid: ", key)

        final_query = ' '

        num_of_lines = 1

        narr = value

        sentences = nltk.sent_tokenize(narr)

        if title_num < len(title_list):

            title = title_list[title_num]

            for line in sentences:

                print('Please wait for 20s...')

                time.sleep(20)

                line_with_title = (line + " " + title).translate(str.maketrans('', '', string.punctuation))
                print("Sentence:", num_of_lines)
                print(line_with_title)

                try:

                    url = 'http://www.google.com/search?'
                    para = {}
                    para["q"] = re.sub("[^\w ]+", " ", line_with_title)
                    para["hl"] = "en"
                    header = {'User-Agent': str(ua.random)}
                    print(header)
                    print("Crawling Page 1...")

                    req = requests.get(url, params=para, headers=header)
                    page = req.text

                except TimeoutError or requests.exceptions.ConnectionError:

                    print("Error occurred")
                    pass

                with open("temp-req", "w") as of:
                    of.write(str(req))

                with open("temp-page", "w") as of:
                    of.write(page)

                soup = BeautifulSoup(page, 'lxml')
                spans = soup.findAll('span', {'class': 'st'})

                lines = [span.get_text() for span in spans]

                snippets = [line.replace('\n', "") for line in lines]

                next_count = 1

                num_of_url = soup.findAll('a', attrs={'class': 'fl'})

                if num_of_url and len(num_of_url) >= 2:
                    while next_count < 3:

                        time.sleep(20)
                        try:
                            print("Crawling Page " + str(next_count + 1) + "...")
                            next_url = 'http://www.google.com' + num_of_url[next_count - 1]['href']
                            next_req = requests.get(next_url)
                            next_page = next_req.text
                            next_soup = BeautifulSoup(next_page, 'lxml')
                            next_spans = next_soup.findAll('span', {'class': 'st'})

                            next_lines = [next_span.get_text() for next_span in next_spans]

                            next_snippets = [line.replace('\n', "") for line in next_lines]
                            snippets.append(next_snippets)

                        except TimeoutError or requests.exceptions.ConnectionError:

                            print("Error occurred")

                            pass

                        next_count += 1

                elif num_of_url and len(num_of_url) < 2:
                    while next_count < len(num_of_url):

                        time.sleep(20)

                        try:
                            print("Crawling Page " + str(next_count + 1) + "...")

                            next_url = 'http://www.google.com' + num_of_url[next_count - 1]['href']
                            next_req = requests.get(next_url)
                            next_page = next_req.text
                            next_soup = BeautifulSoup(next_page, 'lxml')
                            next_spans = next_soup.findAll('span', {'class': 'st'})

                            next_lines = [next_span.get_text() for next_span in next_spans]

                            next_snippets = [line.replace('\n', "") for line in next_lines]
                            snippets.append(next_snippets)

                        except TimeoutError or requests.exceptions.ConnectionError:

                            print("Error occurred")

                            pass

                        next_count += 1

                narr_snippets[key][str(num_of_lines)] = str(snippets)

                print("Start calculating TF-IDF and obtain top k...")
                # mapped_stems = map_stems(str(snippets).lower().translate(str.maketrans('', '', string.punctuation)))

                special_char = ["‘", "’", "·", "–", "“", "”"]
                no_punctuation = str(snippets).translate(str.maketrans('', '', string.punctuation)).translate(
                    {ord(c): 'special char' for c in special_char}).lower()

                mapped_stems = map_stems(no_punctuation)

                # calculate tf-idf score for each term in all searched snippets
                tfidf_dict = {}
                # lowers = str(snippets).lower()
                # no_punctuation = lowers.translate(str.maketrans('', '', string.punctuation))
                tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
                tfs = tfidf.fit_transform([no_punctuation])
                feature_names = tfidf.get_feature_names()
                for col in tfs.nonzero()[1]:
                    tfidf_dict[feature_names[col]] = tfs[0, col]

                # if narr has only one sentence, get top 10 terms
                if len(sentences) == 1:

                    expanded_query = ''
                    sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
                    top_10_elements = sorted_tfidf[:10]
                    for item in top_10_elements:
                        expanded_query = mapped_stems[item[0]] + ' ' + expanded_query

                    final_query = expanded_query + final_query

                # if narr has more than one sentence, get top 5 terms for each sentence
                elif len(sentences) > 1:

                    expanded_query = ''
                    sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
                    top_5_elements = sorted_tfidf[:5]
                    for item in top_5_elements:
                        expanded_query = mapped_stems[item[0]] + ' ' + expanded_query

                    final_query = expanded_query + final_query

                num_of_lines += 1

            narr_queries[key] = final_query

            title_num += 1
        elapsed_time = time.time() - start_time

        print("Time spent: ", elapsed_time)

    with open('narr_snippets.json', 'w', encoding='utf-8') as f:
        json.dump(narr_snippets, f, ensure_ascii=False, indent=1)

    return narr_queries


def get_narr_queries_json(narr_queries):
    narr_queries_json = []
    for key, value in narr_queries.items():
        single = {"topid": key, "title": value}
        narr_queries_json.append(single)

    return narr_queries_json


orignal_query_file = "judged_topics"


def main():
    narr_dict = get_narr_dict(orignal_query_file)

    narr_queries = get_narr_queries(narr_dict)
    narr_queries_json = get_narr_queries_json(narr_queries)

    with open('narr.json', 'w', encoding='utf-8') as f:
        json.dump(narr_queries_json, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()

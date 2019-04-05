import json
import nltk
import string
import time
import requests
from bs4 import BeautifulSoup
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from fake_useragent import UserAgent


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


def get_desc_dict(orignal_query_file):
    desc_dict = {}

    with open(orignal_query_file) as f:
        profiles = json.load(f)
        for prof in profiles:
            desc_dict[prof["topid"]] = prof["description"]
    return desc_dict


def get_desc_queries(desc_dict):
    desc_queries = {}
    desc_snippets = {}

    start_time = time.time()
    ua = UserAgent(cache=True)

    for key, value in desc_dict.items():

        print("Topid: ", key)
        print('Please wait for 20s...')

        time.sleep(20)

        print('Crawling Page 1...')

        try:
            url = 'http://www.google.com/search?'
            para = {}
            para["q"] = re.sub("[^\w ]+", " ", value)
            para["hl"] = "en"
            # print(ua.chrome)
            header = {'User-Agent': str(ua.random)}
            # header = {'User-Agent': str(ua.chrome)}
            print(header)

            req = requests.get(url, params=para, headers=header)
            page = req.text

            soup = BeautifulSoup(page, 'lxml')
            spans = soup.findAll('span', {'class':'st'})
            lines = [span.get_text() for span in spans]
            snippets = [line.replace('\n',"") for line in lines]

        except TimeoutError or requests.exceptions.ConnectionError:

            print("Error occurred")

            pass

        with open("temp-req","w") as of:
            of.write(str(req))

        with open("temp-page","w") as of:
            of.write(page)

        next_count = 1

        num_of_url = soup.findAll('a', attrs={'class': 'fl'})

        if num_of_url and len(num_of_url) >= 2:
            while next_count < 3:
            # while next_count < 5:
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

        desc_snippets[key] = str(snippets)
        # calculate tf-idf score for each term in all searched snippets
        print("Start calculating TF-IDF and obtain top 10...")
        # mapped_stems = map_stems(str(snippets).lower().translate(str.maketrans('', '', string.punctuation)))

        special_char = ["‘", "’", "·", "–", "“", "”"]
        no_punctuation = str(snippets).translate(str.maketrans('', '', string.punctuation)).translate(
            {ord(c): 'special char' for c in special_char}).lower()

        mapped_stems = map_stems(no_punctuation)
        tfidf_dict = {}
        # lowers = str(snippets).lower()
        # no_punctuation = lowers.translate(str.maketrans('', '', string.punctuation))
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        tfs = tfidf.fit_transform([no_punctuation])
        feature_names = tfidf.get_feature_names()
        for col in tfs.nonzero()[1]:
            tfidf_dict[feature_names[col]] = tfs[0, col]

        # get top 10 desc terms as expanded query

        expanded_query = ''
        sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
        top_10_elements = sorted_tfidf[:10]
        for item in top_10_elements:
            expanded_query = mapped_stems[item[0]] + ' ' + expanded_query

        desc_queries[key] = expanded_query

        elapsed_time = time.time() - start_time

        print("Time spent: ", elapsed_time)
    with open('desc_snippets.json', 'w', encoding='utf-8') as f:
        json.dump(desc_snippets, f, ensure_ascii=False, indent=1)

    return desc_queries


def get_desc_queries_json(desc_queries):

    desc_queries_json = []
    for key, value in desc_queries.items():
        single = {"topid": key, "title": value}
        desc_queries_json.append(single)

    return desc_queries_json


orignal_query_file = "judged_topics"


def main():
    desc_dict = get_desc_dict(orignal_query_file)
    desc_queries = get_desc_queries(desc_dict)
    desc_queries_json = get_desc_queries_json(desc_queries)

    with open('desc.json', 'w', encoding='utf-8') as f:
        json.dump(desc_queries_json, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
        main()


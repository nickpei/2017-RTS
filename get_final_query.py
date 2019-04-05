import json
import string

orignal_query_file = "judged_topics"


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


def assign_weight(dict):
    weighted = {}
    i = 0
    with open(orignal_query_file) as f:
        profiles = json.load(f)

    for key, value in dict.items():
        ulist = unique_list(value.split())

        # title_tuple = tuple(value.translate(str.maketrans('', '', string.punctuation)).split())
        for term in ulist:
            if term in profiles[i]['title'].lower():
                weighted[term] = 1
            elif(term not in profiles[i]['title'].lower()) and (term not in profiles[i]['description'].lower()) \
                    and (term not in profiles[i]['narrative'].lower()):
                weighted[term] = 0.2
            else:
                weighted[term] = 0.3
        dict[key] = weighted
        weighted = {}
        i += 1

    return dict

def combine_query(title_query, desc_query, narr_query):

    title_dict = {}
    desc_dict = {}
    narr_dict = {}
    final_dict = {}
    with open(title_query) as f:
        titles = json.load(f)
        for title in titles:
            title_dict[title["topid"]] = title["title"]

    with open(desc_query) as f:
        descs = json.load(f)
        for desc in descs:
            desc_dict[desc["topid"]] = desc["title"]

    with open(narr_query) as f:
        narrs = json.load(f)
        for narr in narrs:
            narr_dict[narr["topid"]] = narr["title"]

    for key, value in title_dict.items():
        final_dict[key] = value + desc_dict[key] + narr_dict[key]

    return assign_weight(final_dict)


def get_final_queries_json(final_dict):
    final_queries_json = []
    for key, value in final_dict.items():
        single = {"topid": key, "title": value}
        final_queries_json.append(single)

    return final_queries_json


def main():

    final_dict = combine_query("title.json", "desc.json", "narr.json")
    final_queries_json = get_final_queries_json(final_dict)

    with open('final_query.json', 'w', encoding='utf-8') as f:
        json.dump(final_queries_json, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()



from bs4 import BeautifulSoup
import requests
from collections import namedtuple

import falconn
import fastText
import numpy as np
import pandas as pd

WIKI_ARTICLE_URL = 'https://ru.wikipedia.org/wiki?curid={pageid}'
WIKI_API_ENDPOINT = 'https://ru.wikipedia.org/w/api.php'
Article = namedtuple('Article', ['title', 'pageid'])


def get_title_by_url(article_url):
    r = requests.get(article_url)
    return BeautifulSoup(r.text, 'lxml').find('h1').get_text() if r.ok else None


def get_wiki_article_summary(pageid):
    params = {
        'action': 'query',
        'prop': 'extracts|pageprops',
        'inprop': 'url',
        'ppprop': 'disambiguation',
        'exintro': True,
        'exlimit': 1,
        'format': 'json',
        'pageids': pageid
    }
    response = requests.get(WIKI_API_ENDPOINT, params=params)
    response.raise_for_status()
    response = response.json()

    page = response['query']['pages'][pageid]
    if 'pageprops' in page:
        return None

    html = response['query']['pages'][pageid].get('extract')
    if not html:
        return None

    first_p = None
    for p in BeautifulSoup(html, 'lxml').find_all('p'):
        r = p.get_text().strip()
        if r:
            first_p = r
            break

    return first_p.replace('́', '') if first_p else None


def search_wiki_article(query):
    params = {
        'action': 'query',
        'srlimit': 1,
        'list': 'search',
        'format': 'json',
        'srinfo': 'totalhits',
        'srprop': '',
        'srsearch': query
    }
    response = requests.get(WIKI_API_ENDPOINT, params=params)
    response.raise_for_status()
    response = response.json()
    data = response['query']['search']
    if data:
        return data[0]['title'], str(data[0]['pageid'])
    else:
        return None


class WikipediaDataSet:
    def __init__(self, meta_info, vectorized_articles, fast_text_vectors):
        self.vectorized_articles = np.load(vectorized_articles)
        self.meta_info = pd.read_csv(meta_info)
        self.table = None
        self.query = None
        self.fast_text = fastText.load_model(fast_text_vectors)

    def build_LSH_index(self):
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = self.vectorized_articles.shape[1]
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
        params_cp.l = 200
        params_cp.num_rotations = 1
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

        falconn.compute_number_of_hash_functions(21, params_cp)
        self.table = falconn.LSHIndex(params_cp)
        self.table.setup(self.vectorized_articles)

        self.query = self.table.construct_query_object()
        self.query.set_num_probes(params_cp.l)

    def find_k_nearest_neighbors(self, title, k=5):
        q = self.fast_text.get_sentence_vector(title)
        ids = self.query.find_k_nearest_neighbors(q, k=k)
        found = self.meta_info.iloc[ids].to_dict(orient='records')
        result = []
        for item in found:
            if item['title'].lower() != title.lower():
                result.append(Article(title=item['title'], pageid=item['id']))
        return result[:3]
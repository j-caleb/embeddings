import sys
sys.path.append('../')

from utils import vector_utils
from utils import commons
import pickle
import numpy as np

class QueryVectors:
    def __init__(self, path):
        with open(path, "rb") as input_file:
            self.vectors= pickle.load(input_file)
            self.vector_len = len(self.vectors[list(self.vectors.keys())[0]])
    def get_similar(self, query, top_n=10):
        query = query.strip()
        query = query.split()
        query = [item for item in query if item in self.vectors]
        if not len(query):
            return None
        query_vec = np.zeros(self.vector_len)
        for term in query:
            query_vec+=self.vectors[term]
        query_vec=query_vec/len(query)
        results = {} # If you train this over a very big corpus brute force cosine is NOT a good idea
        for term in self.vectors:
            score = vector_utils.cosine_similarity(query_vec, self.vectors[term])
            results[term]=score
        results = commons.sort_dictionary(results)
        return results[0:top_n]

"""
This is an implementation of Random Indexing based on the paper Introduction to
Random Indexing by Magnus Sahlgren. Random Indexing relies upon random projection
to map objects (terms in a document, vertices in a network, etc.) to fixed
dimensional vectors. This is a method for generating embeddings similar to word2vec.
"""

import numpy as np
from collections import defaultdict
import math
import random

def compute_idf(data, min_count):
    """
    IDF is used to weight the term vectors.
    """
    counts = defaultdict(float)
    for line in data:
        line = set(line.split())
        for feature in line:
            counts[feature]+=1
    delete = [feature for feature in counts if counts[feature] < min_count]
    for feature in delete:
        del counts[feature]
    for feature in counts:
        counts[feature]=math.sqrt(len(data)/counts[feature])
    return counts

def initialize_vectors(features, idf, dim, seeds):
    """
    This creates the initial random projection for each feature. You create initial
    vector with dimensionality dim. Dim should be in the range 500-1000. You then
    select n (n is determined by seeds) elements and set the value to 1 or -1
    randomly. This performs the random projection.
    """
    vectors = {}
    for feature in features:
        vector=np.zeros(dim)
        sample=random.sample(dim,seeds) # Grab the n random elements for random projection
        for index in sample:
            vector[index]=random.choice([-1.0,1.0]) # Set each element to +1 or -1 for random projection
        vector=vector * idf[feature] # Weight based on IDF
        vectors[feature]=vector
    return vectors

def train_vectors(data, vectors):

def save():

def get_data(path):
    f = open(path,'r')
    return f.read().split('\n')

def train(in_file, out_dir, save_name='ri_index', seeds=20, dim=500, min_count=10):
    """
    in_file = File containing co-occurrence data. Expected format is one line
    per entry with a space between each item. This can be anything from citations
    from articles to text. Also, this is expecting that tokenization and all other
    necessary cleanup has been performed.

    out_dir = Location to store the results.

    save_name = Optional name of the trained vectors.

    seeds = The number of random seeds used for random projection. The minimum
    should be 10.

    dim = Dimensionality for the term vectors. Should be 500-1000 (or greater)

    min_count = Threshold for building vectors. Anything less than min_count is discarded.
    """

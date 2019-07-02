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
import pickle
import argparse

import sys
sys.path.append('../')
from utils import commons
from utils import store
from utils import vector_utils

print_every = 100000
print_status = True

def compute_idf(data, min_count):
    """
    IDF is used to weight the term vectors.
    """
    if print_status:
        print('Computing IDF')
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

    for i in range(len(features)):
        if print_status and i % print_every == 0:
            print('Initializing ' + str(i))
        feature = features[i]
        vector=np.zeros(dim)
        sample=random.sample(range(0,dim),seeds) # Grab the n random elements for random projection
        for index in sample:
            vector[index]=random.choice([-1.0,1.0]) # Set each element to +1 or -1 for random projection
        vector=vector * idf[feature] # Weight based on IDF
        vectors[feature]=vector
    return vectors

def train_vectors(data, vectors):
    """
    For each feature in each line, add the feature to all other features. Conceptually,
    each co-occurance of two features moves the two features closer together.
    """
    trained_vectors=vectors.copy()
    for i in range(len(data)):
        if print_status and i % print_every == 0:
            print('Processed ' + str(i))
        line = data[i].split()
        line=[feature for feature in line if feature in vectors]
        for feature_1 in line:
            for feature_2 in line:
                if feature_1 != feature_2:
                    trained_vectors[feature_2]+=vectors[feature_1]
        for feature in trained_vectors:
            trained_vectors[feature] = vector_utils.normalize_vector(trained_vectors[feature])
        return trained_vectors

def train(in_file, out_dir, file_name='ri_index', seeds=20, dim=500, min_count=10):
    data=commons.get_data(in_file)
    idf=compute_idf(data, min_count)
    vectors = initialize_vectors(list(idf.keys()), idf, dim, seeds)

    for i in range(2):
        vectors = train_vectors(data, vectors) # Performs two training cycles

    store.pickle_dict(vectors, out_dir, file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in','--in_file', help='Input director and file name.', required=False, default='')
    parser.add_argument('-out','--out_dir', help='Location to store the results.', required=False, default='')
    parser.add_argument('-name','--file_name', help='Name for the index when storing.', required=False, default='ri_index')
    parser.add_argument('-s','--seeds', help='Number of seeds for random projection indexing. Should be 10-50.', required=False, default=20)
    parser.add_argument('-d','--dim', help='Number of dimensions for vectors. Range should be 500-1000', required=False, default=500)
    parser.add_argument('-min','--min_count', help='Minimum frequency of occurance threshold.', required=False, default=10)
    parser.add_argument('-p','--print_status', help='Print progress during execution. False is off. Default is True', required=False, default=True)
    parser.add_argument('-pe','--print_every', help='How often to print status during exectuation. Default is every 500k lines.', required=False, default=500000)

    args = vars(parser.parse_args())

    print_every = args['print_every']
    print_status = args['print_status']

    if args['in_file'] == '': # If you execute with no arguments it will defualt to using a config file
        from config_files.ri_config import  config
        train(config['in_file'],config['out_dir'])
    else:
        train(args['in_file'], args['out_dir'], file_name=args['file_name'], seeds=args['seeds'], dim=args['dim'], min_count=args['min_count'])

"""
This is an implementation of Random Indexing based on the paper Introduction to
Random Indexing by Magnus Sahlgren. Random Indexing relies upon random projection
to map objects (terms in a document, vertices in a network, etc.) to fixed
dimensional vectors. This is a method for generating embeddings similar to word2vec.
"""

import numpy as np
import math
import random
import pickle
import argparse

import sys
sys.path.append('../')
from utils import commons
from utils import vector_utils
from utils import text_utils

print_every = 100000
print_status = True

def initialize_vectors(valid_tokens, dim, seeds):
    """
    This creates the initial random projection for each feature. You create initial
    vector with dimensionality dim. Dim should be in the range 500-1000. You then
    select n (n is determined by seeds) elements and set the value to 1 or -1
    randomly. This performs the random projection.
    """
    vectors = {}

    for i in range(len(valid_tokens)):
        if print_status and i % print_every == 0:
            print('Initializing ' + str(i))
        token = valid_tokens[i]
        vector=np.zeros(dim)
        sample=random.sample(range(0,dim),seeds) # Grab the n random elements for random projection
        for index in sample:
            vector[index]=random.choice([-1.0,1.0]) # Set each element to +1 or -1 for random projection
        vectors[token]=vector
    return vectors

def train_vectors(data, vectors):
    """
    For each feature in each line, add the feature to all other features. Conceptually,
    each co-occurance of two features moves the two features closer together.
    """
    trained_vectors=copy.deepcopy(vectors)
    for i in range(len(data)):
        if print_status and i % print_every == 0:
            print('Processed ' + str(i))
        line = data[i].split()
        line=[token for token in line if token in vectors]
        for token_1 in line: # This is it for the training! Simple addition.
            for token_2 in line:
                if token_1 != token_2:
                    trained_vectors[token_1]+=vectors[token_2]

    for token in trained_vectors:
        trained_vectors[token] = vector_utils.normalize_vector(trained_vectors[token])
    return trained_vectors

def train_vectors_window(data, vectors, window_size):
    """
    Here training is performed using a context window. Context window should be >= 5.
    """
    trained_vectors=copy.deepcopy(vectors)
    valid_tokens = set(vectors.keys())
    for i in range(len(data)):
        if print_status and i % print_every == 0:
            print('Processed ' + str(i))
        line=data[i]
        training = create_context_training(line, widow_size, valid_tokens)
        for example in training:
            target=example[0]
            context=example[1]
            for token in context: # The training is simply adding the vectors for the tokens in the context window to the target vector
                trained_vectors[token]+=vectors[token]
    for token in trained_vectors:
        trained_vectors[token] = vector_utils.normalize_vector(trained_vectors[token])
    return trained_vectors

def train(in_file, out_dir, file_name='ri_index', seeds=20, dim=500, min_count=10, window_size=None):
    data=commons.get_data(in_file)
    valid_terms = text_utils(data)
    if print_status:
        print('Building ' + str(len(valid_terms)) + ' term vectors')
    vectors = initialize_vectors(valid_terms, dim, seeds)

    for i in range(2):
        if window_size is None:
            vectors = train_vectors(data, vectors) # Performs two training cycles
        else:
            vectors = train_vectors_window(data, vectors, window_size)

    commons.pickle_dict(vectors, out_dir, file_name)

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
    parser.add_argument('-w','--window_size', help='If this is not set to None it will trigger training based on context window. Use at least 5 if you want to do this.', required=False, default=None)

    args = vars(parser.parse_args())

    print_every = args['print_every']
    print_status = args['print_status']

    if args['in_file'] == '': # If you execute with no arguments it will defualt to using a config file
        from config_files.ri_config import  config
        print_status = config['print_status']
        print_every=500000
        train(config['in_file'], config['out_dir'], file_name=config['file_name'], seeds=config['seeds'], dim=config['dim'], min_count=config['min_count'], window_size=config['window_size'])
    else:
        train(args['in_file'], args['out_dir'], file_name=args['file_name'], seeds=args['seeds'], dim=args['dim'], min_count=args['min_count'], window_size=args['window_size'])

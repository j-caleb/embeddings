"""
This generates embeddings using random projection. This method can associate two
different spaces of information. For example, you could associate terms with authors.
Then you could find the terms most strongly associated with an author.
"""

import numpy as np
import math
import random
import pickle
import argparse
import json
from collections import defaultdict

import sys
sys.path.append('../')
from utils import commons
from utils import vector_utils
from utils import text_utils

print_every = 100000
print_status = True

def get_valid_metadata(data, min_count):
    counts = defaultdict(float)
    for line in data:
        for item in line:
            counts[item]+=1
    remove = [item for item in counts if counts[item] <= min_count]
    for item in remove:
        del counts[item]
    return list(counts.keys())

def get_valid_features(data, field, min_count):
    data_field = [item[field] for item in data]
    if isinstance(data_a[0],str):
        return text_utils.get_valid_terms(data_field, min_count)
    else:
        return get_valid_metadata(data_field, min_count)

def remove_invalid(line, vectors):
    if isinstance(line, str):
        line = line.split()
    line = [el for el in line if el in vectors]
    return line

def train_vectors(data, source, target, source_vectors, target_vectors):
    """
    Performs one training step. This adds from source_vectors to target_vectors.
    Given data[source]=[a,b] and target_vectors[target]=[c], the following is performed:
        target_vectors[c]+=source_vectors[a]
        target_vectors[c]+=source_vectors[b]
    Conceptually, the vectors for source and target move closer together the
    more they co-occur.
    """
    for line in data:
        source_data = remove_invalid(line[source])
        target_data = remove_invalid(line[target])
        for s_feature in source_data:
            for t_feature in target_data:
                target_vectors[t_feature]+=source_vectors[s_feature]
    for feature in target_vectors:
        target_vectors[feature]=vector_utils.normalize_vector(target_vectors[feature])
    return target_vectors


def train(in_file, out_dir, field_a, field_b, file_name='meta2vec', seeds=20, dim=500, min_count=10, window_size=None):
    """
    The input file is expected to be JSON. field_a will have random projection run on it.
    I am assuming that field_a and field_b are a string or a list. If the type is a list,
    no stopwords will be removed. The assumption is that this is a metadata field.
    """
    data=commons.get_data(in_file)
    data = [json.loads(item) for item in data]

    valid_features_a = get_valid_features(data, field_a, min_count)
    valid_features_b = get_valid_features(data, field_b, min_count)

    if print_status:
        print('Building ' + str(len(valid_features_a)) + ' vectors for ' + field_a)
        print('Building ' + str(len(valid_features_b)) + ' vectors for ' + field_b + '\n')

    vectors_a = vector_utils.initialize_vectors_random_projection(valid_features_a, dim, seeds)
    vectors_b = vector_utils.initialize_vectors_random_projection(valid_features_b, dim, seeds)

    vectors_b = train_vectors(data, field_a, field_b, vectors_a, vectors_b) # Add a to b
    vectors_a = train_vectors(data, field_b, field_a, vectors_b, vectors_a) # Add b to a
    vectors_b = train_vectors(data, field_a, field_b, vectors_a, vectors_b) # Add a to b

    commons.pickle_dict(vectors_a, out_dir, file_name+'_'+field_a)
    commons.pickle_dict(vectors_b, out_dir, file_name+'_'+field_b)

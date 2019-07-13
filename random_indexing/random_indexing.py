"""
This is an implementation of Random Indexing based on the papers Introduction to
Random Indexing by Magnus Sahlgren and Reflection Random Indexing and indirect
inference by Cohen, Schvaneveldt, and Widdows. Random Indexing relies upon random
projection to map objects (terms in a document, vertices in a network, etc.) to fixed
dimensional vectors. This is a method for generating embeddings similar to word2vec.
"""

import numpy as np
import math
import random
import pickle
import argparse
import copy

import sys
sys.path.append('../')
from utils import commons
from utils import vector_utils
from utils import text_utils

def add_terms_to_doc(documents, text_field, filename_field, term_vectors, doc_vectors):
    for doc in documents:
        doc_vec = doc_vectors[doc[filename_field]]
        for term in doc[text_field]:
            doc_vec+=term_vectors[term]
    for file_name in doc_vectors:
        doc_vectors[file_name] = vector_utils.normalize_vector(doc_vectors[file_name])

def add_doc_to_terms(documents, text_field, filename_field, term_vectors, doc_vectors):
    for doc in documents:
        doc_vec = doc_vectors[doc[filename_field]]
        for term in doc[text_field]:
            term_vectors[term]+=doc_vec
    for term in term_vectors:
        term_vectors[term] = vector_utils.normalize_vector(term_vectors[term])

def train_vectors_ri(documents, text_field, filename_field, min_count, dim, seeds, file_names, valid_terms):
    """
    This is the original Random Indexing method. You initialize the document
    vectors using random projection. You initialize the term vectors to all zeros.
    When a term occurs in the document, add the document vector to the term vector.
    This will generate useful term vectors, but the document vectors are not useful
    as they are simply random projections.
    """
    term_vectors = vector_utils.initialize_vectors_zeros(valid_terms, dim)
    doc_vectors = vector_utils.initialize_vectors_random_projection(file_names, dim, seeds)
    add_doc_to_terms(documents, text_field, filename_field, term_vectors, doc_vectors)
    return term_vectors

def train_vectors_trri(documents, text_field, filename_field, min_count, dim, seeds, file_names, valid_terms):
    """
    This is term based Random Indexing with extra training cycles similar to drri.
    Initialize the term vectors with random projection. Generate document vectors by adding
    all of the terms vectors for the terms in the document. For each term, add the
    document vector where it occurs to its term vector. This will generate useful
    term vectors and document vectors.
    """
    term_vectors = vector_utils.initialize_vectors_random_projection(valid_terms, dim, seeds)
    doc_vectors = vector_utils.initialize_vectors_zeros(file_names, dim)
    add_terms_to_doc(documents, text_field, filename_field, term_vectors, doc_vectors)
    add_doc_to_terms(documents, text_field, filename_field, term_vectors, doc_vectors)
    add_terms_to_doc(documents, text_field, filename_field, term_vectors, doc_vectors)
    return term_vectors, doc_vectors

def train_vectors_drri(documents, text_field, filename_field, min_count, dim, seeds, file_names, valid_terms):
    """
    This is document based Random Indexing. You initialize the document vectors
    using random projection. The term vectors are initialized to zeros. When a term
    occurrs in the document, add the document vector to the term vector. Normalize the
    term vectors. In the next step, for each document, add the term vectors to the document
    vector. Normalize the document vectors. Finally, for each term add the document vector of
    each document in which it appears.
    """
    doc_vectors = vector_utils.initialize_vectors_random_projection(file_names, dim, seeds)
    term_vectors = vector_utils.initialize_vectors_zeros(valid_terms, dim)
    add_doc_to_terms(documents, text_field, filename_field, term_vectors, doc_vectors)
    add_terms_to_doc(documents, text_field, filename_field, term_vectors, doc_vectors)
    add_doc_to_terms(documents, text_field, filename_field, term_vectors, doc_vectors)
    add_terms_to_doc(documents, text_field, filename_field, term_vectors, doc_vectors)
    return term_vectors, doc_vectors

def train_vectors_sliding_window(documents, text_field, valid_terms, window_size, dim, seeds):
    """
    Training is performed using a context window. For each term, grab the neighbors
    before and after the term. The width is determined by window_size. Add all of
    the vectors for the terms in the context window and add these to the target
    term. The motivation is that for large documents it may not be advisable to
    treat the entire document as context.
    """
    print('training sliding window')
    vectors = vector_utils.initialize_vectors_random_projection(valid_terms, dim, seeds)
    trained_vectors = copy.deepcopy(vectors)
    for i in range(len(documents)):
        if i % 1000 == 0:
            print(i)
        training = text_utils.create_context_training(documents[i][text_field], window_size, valid_terms) # For each term in the document generate a context window
        for example in training:
            target = example[0]
            context = example[1]
            for term in context: # The training is simply adding the vectors for the tokens in the context window to the vector for the target token
                trained_vectors[target]+=vectors[term]
    for term in trained_vectors:
        trained_vectors[term] = vector_utils.normalize_vector(trained_vectors[term])
    return trained_vectors

def train_vectors_metadata():
    return True

def get_valid_terms(documents, min_count, text_field):
    if type(documents[0][text_field]) is list:
        for doc in documents:
            doc[text_field]=' '.join(doc[text_field])
    text = [doc[text_field] for doc in documents]
    return text_utils.get_valid_terms(text, min_count)

def clean_documents(documents, valid_terms, text_field):
    """
    Splitting operations and valid term checks can happen many times depending
    upon the mode. This just performs this step once.
    """
    for doc in documents:
        text = doc[text_field]
        text = [term for term in text.split() if term in valid_terms]
        doc[text_field]=text
    return documents

def train(in_file, out_dir, file_name='ri_index', seeds=20, dim=500, min_count=10, window_size=None, sample=None, text_field='text', filename_field='file_name', mode='ri'):
    documents = commons.get_data(in_file)

    if sample is not None:
        random.shuffle(documents)
        documents = documents[0:sample]

    valid_terms = get_valid_terms(documents, min_count, text_field)
    documents = clean_documents(documents, valid_terms, text_field)
    file_names = [doc[filename_field] for doc in documents]

    term_vectors = None
    doc_vectors = None

    if mode == 'metadata':
        train_vectors_metadata()
    elif mode == 'window':
        term_vectors = train_vectors_sliding_window(documents, text_field, valid_terms, window_size, dim, seeds)
    elif mode == 'ri':
        term_vectors = train_vectors_ri(documents, text_field, filename_field, min_count, dim, seeds, file_names, valid_terms)
    elif mode == 'trri':
        term_vectors, doc_vectors = train_vectors_trri(documents, text_field, filename_field, min_count, dim, seeds, file_names, valid_terms)
    elif mode == 'drri':
        term_vectors, doc_vectors = train_vectors_drri(documents, text_field, filename_field, min_count, dim, seeds, file_names, valid_terms)

    commons.pickle_dict(term_vectors, out_dir, file_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in','--in_file', help='Input directory and file name. Expected format is one sentence per line. The assumption is that you have already cleaned the sentences.', required=False, default='')
    parser.add_argument('-out','--out_dir', help='Location to store the results.', required=False, default='')
    parser.add_argument('-name','--file_name', help='Name for the index when storing.', required=False, default='ri_index')
    parser.add_argument('-s','--seeds', help='Number of seeds for random projection indexing. Should be 10-50.', required=False, default=20)
    parser.add_argument('-d','--dim', help='Number of dimensions for vectors. Range should be 500-1000', required=False, default=500)
    parser.add_argument('-min','--min_count', help='Minimum frequency of occurance threshold.', required=False, default=10)
    parser.add_argument('-p','--print_status', help='Print progress during execution. False is off. Default is True', required=False, default=True)
    parser.add_argument('-pe','--print_every', help='How often to print status during exectuation. Default is every 500k lines.', required=False, default=500000)
    parser.add_argument('-w','--window_size', help='If this is not set to None it will trigger training based on context window. Use at least 5 if you want to do this.', required=False, default=None)
    parser.add_argument('-m','--mode', help='Determines the method for generating vectors. ri=Random Indexing, trri=term based Reflective Random Indexing, drri=document based Reflective Random Indexing, window=Random Indexing with sliding window, meta=Reflective Random Indexing with metadata', required=False, default=500000)
    parser.add_argument('-tx','--text_field', help='Name of the field that contains the text to be proecessed.', required=False, default=None)
    parser.add_argument('-fn','--filename_field', help='Name of the field that contains the unique identifier for a document.', required=False, default=None)

    args = vars(parser.parse_args())

    if args['in_file'] == '': # If you execute with no arguments it will defualt to using a config file
        from data.config_files import ri_config
        config = ri_config.config
        print_status = config['print_status']
        print_every = config['print_every']
        train(config['in_file'], config['out_dir'], file_name=config['file_name'], seeds=config['seeds'], dim=config['dim'], min_count=config['min_count'], window_size=config['window_size'], sample=config['sample'], text_field=config['text_field'], filename_field=config['filename_field'], mode=config['mode'])
    else:
        print_status = args['print_status']
        print_every = args['print_every']
        train(args['in_file'], args['out_dir'], file_name=args['file_name'], seeds=args['seeds'], dim=args['dim'], min_count=args['min_count'], window_size=args['window_size'], sample=args['sample'])

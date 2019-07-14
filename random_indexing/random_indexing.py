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
from collections import defaultdict

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

def add_terms_to_meta(documents, text_field, metadata_field, term_vectors, meta_vectors):
    for doc in documents:
        for term in doc[text_field]:
            for meta in doc[metadata_field]:
                meta_vectors[meta]+=term_vectors[term]
    for meta in meta_vectors:
        meta_vectors[meta] = vector_utils.normalize_vector(meta_vectors[meta])

def add_meta_to_terms(documents, text_field, metadata_field, term_vectors, meta_vectors):
    for doc in documents:
        for meta in doc[metadata_field]:
            for term in doc[text_field]:
                term_vectors[term]+=meta_vectors[meta]
    for term in term_vectors:
        term_vectors[term] = vector_utils.normalize_vector(term_vectors[term])

def train_vectors_ri(documents, text_field, filename_field, dim, seeds, file_names, valid_terms):
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

def train_vectors_trri(documents, text_field, filename_field, dim, seeds, file_names, valid_terms):
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

def train_vectors_drri(documents, text_field, filename_field, dim, seeds, file_names, valid_terms):
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
    vectors = vector_utils.initialize_vectors_random_projection(valid_terms, dim, seeds)
    trained_vectors = copy.deepcopy(vectors)
    for i in range(len(documents)):
        training = text_utils.create_context_training(documents[i][text_field], window_size, valid_terms) # For each term in the document generate a context window
        for example in training:
            target = example[0]
            context = example[1]
            for term in context: # The training is simply adding the vectors for the tokens in the context window to the vector for the target token
                trained_vectors[target]+=vectors[term]
    for term in trained_vectors:
        trained_vectors[term] = vector_utils.normalize_vector(trained_vectors[term])
    return trained_vectors

def train_vectors_metadata(documents, text_field, metadata_field, dim, seeds, valid_terms, meta_min_count):
    """
    This implementation is associating text with some form of metadata. The assumption
    is that the metadata is in a list. This implementation is similar to DRRI, but is
    performed on text & metadata.
    """
    lbl_cnt = defaultdict(float) # Filtering metadata. Not happy with this. Need to fix text_utils to handle this
    for doc in documents:
        for lbl in doc[metadata_field]:
            lbl_cnt[lbl]+=1
    remove = [lbl for lbl in lbl_cnt if lbl_cnt[lbl] <= meta_min_count]
    for lbl in remove:
        del lbl_cnt[lbl]
    for doc in documents:
        doc[metadata_field]=[lbl for lbl in doc[metadata_field] if lbl in lbl_cnt]

    meta_vectors = vector_utils.initialize_vectors_random_projection(list(lbl_cnt.keys()), dim, seeds)
    term_vectors = vector_utils.initialize_vectors_zeros(valid_terms, dim)
    add_meta_to_terms(documents, text_field, metadata_field, term_vectors, meta_vectors)
    add_terms_to_meta(documents, text_field, metadata_field, term_vectors, meta_vectors)
    add_meta_to_terms(documents, text_field, metadata_field, term_vectors, meta_vectors)
    add_terms_to_meta(documents, text_field, metadata_field, term_vectors, meta_vectors)
    return term_vectors, meta_vectors

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

def train(in_file, out_dir, seeds=20, dim=500, min_count=10, window_size=None, sample=None, text_field='text', filename_field='file_name', mode='ri', metadata_field='labels', meta_min_count=10):
    """
    Note: Currently I am not saving the document vectors. To make this practical I need to add locality sensitive hashing for fast nearest neighbor search.
    """
    documents = commons.get_data(in_file)

    if sample is not None:
        random.shuffle(documents)
        documents = documents[0:sample]

    valid_terms = get_valid_terms(documents, min_count, text_field)
    documents = clean_documents(documents, valid_terms, text_field)
    file_names = [doc[filename_field] for doc in documents]

    term_vectors = None
    doc_vectors = None
    file_name = None

    if mode == 'window':
        term_vectors = train_vectors_sliding_window(documents, text_field, valid_terms, window_size, dim, seeds)
        file_name = 'term_index_window'
    elif mode == 'ri':
        term_vectors = train_vectors_ri(documents, text_field, filename_field, dim, seeds, file_names, valid_terms)
        file_name = 'term_index_ri'
    elif mode == 'trri':
        term_vectors, doc_vectors = train_vectors_trri(documents, text_field, filename_field, dim, seeds, file_names, valid_terms)
        file_name = 'term_index_trri'
    elif mode == 'drri':
        term_vectors, doc_vectors = train_vectors_drri(documents, text_field, filename_field, dim, seeds, file_names, valid_terms)
        file_name = 'term_index_drri'
    else:
        print(mode + ' is not a recognized option.')
        return

    commons.pickle_dict(term_vectors, out_dir, file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in','--in_file', help='Input directory and file name. Expected format is one sentence per line. The assumption is that you have already cleaned the sentences.', required=False, default='')
    parser.add_argument('-out','--out_dir', help='Location to store the results.', required=False, default='')
    parser.add_argument('-s','--seeds', help='Number of seeds for random projection indexing. Should be 10-50.', required=False, default=20)
    parser.add_argument('-d','--dim', help='Number of dimensions for vectors. Range should be 500-1000', required=False, default=500)
    parser.add_argument('-min','--min_count', help='Minimum frequency of occurance threshold.', required=False, default=10)
    parser.add_argument('-w','--window_size', help='If this is not set to None it will trigger training based on context window. Use at least 5 if you want to do this.', required=False, default=None)
    parser.add_argument('-m','--mode', help='Determines the method for generating vectors. ri=Random Indexing, trri=term based Reflective Random Indexing, drri=document based Reflective Random Indexing, window=Random Indexing with sliding window', required=False, default=500000)
    parser.add_argument('-tx','--text_field', help='Name of the field that contains the text to be proecessed.', required=False, default=None)
    parser.add_argument('-fn','--filename_field', help='Name of the field that contains the unique identifier for a document.', required=False, default=None)
    parser.add_argument('-meta','--metadata_field', help='Name of metadata field. Metadata must be a list.', required=False, default=None)
    parser.add_argument('-mmin','--meta_min_count', help='Minimum count for metadata.', required=False, default=None)

    args = vars(parser.parse_args())

    if args['in_file'] == '': # If you execute with no arguments it will defualt to using a config file
        from data.config_files import ri_config
        config = ri_config.config

        (train(config['in_file'],
            config['out_dir'],
            seeds=config['seeds'],
            dim=config['dim'],
            min_count=config['min_count'],
            window_size=config['window_size'],
            sample=config['sample'],
            text_field=config['text_field'],
            filename_field=config['filename_field'],
            mode=config['mode'],
            metadata_field=config['metadata_field'],
            meta_min_count=config['meta_min_count']))
    else:
        (train(args['in_file'],
            args['out_dir'],
            seeds=args['seeds'],
            dim=args['dim'],
            min_count=args['min_count'],
            window_size=args['window_size'],
            sample=args['sample'],
            text_field=args['text_field'],
            filename_field=args['filename_field'],
            mode=args['mode'],
            metadata_field=args['metadata_field'],
            meta_min_count=args['meta_min_count']))

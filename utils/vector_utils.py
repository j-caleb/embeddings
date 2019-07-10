import numpy as np
from scipy.spatial.distance import cosine

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.array([0.] * len(vector))
    return np.array(vector) / norm

def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.
    return 1. - cosine(v1, v2)

def initialize_vectors_random_projection(features, dim, seeds):
    """
    This creates the initial random projection for each feature. You create initial
    vector with dimensionality dim. Dim should be in the range 500-1000. You then
    select n (n is determined by seeds) elements and set the value to 1 or -1
    randomly. This performs the random projection.
    """
    vectors = {}

    features = list(features)

    for i in range(len(features)):
        if print_status and i % print_every == 0:
            print('Initializing ' + str(i))
        feature = features[i]
        vector=np.zeros(dim)
        sample=random.sample(range(0,dim),seeds) # Grab the n random elements for random projection
        for index in sample:
            vector[index]=random.choice([-1.0,1.0]) # Set each element to +1 or -1 for random projection
        vectors[feature]=vector
    return vectors

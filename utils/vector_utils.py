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

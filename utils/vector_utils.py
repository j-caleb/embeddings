import numpy as np

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.array([0.] * len(vector))
    return np.array(vector) / norm

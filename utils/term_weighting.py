import math
from collections import defaultdict
import text_utils

def compute_idf(counts, n_docs):
    idf = {}
    for token in counts:
        idf[token]=math.sqrt(n_docs/counts[token])
    return idf

# def compute_log_entropy(data, min_count):
#     """
#     Log entropy has a history of being used in distributional semantics (LSA).
#     Instead of using term frequency directly, training is done using local statistics
#     (i.e. current document) and global statistics (i.e. all documents). What I am
#     doing here is computing all log entropy for all documents.
#
#     This needs cleanup.
#     """
#     global_frequency = defaultdict(float)
#     local_frequency = []
#     for line in data: # Grabbing both the local and global term frequency counts
#         line = text_utils.strip_stopwords(line).split()
#         for token in set(line):
#             global_frequency[token]+=1
#         term_frequency = defaultdict(float)
#         for token in line:
#             term_frequency[token]+=1
#         local_frequency.append(term_frequency)
#     remove = [token for token in global_frequency if global_frequency[token] <= min_count]
#     for token in remove:
#         del global_frequency[token]
#     log_entropy=[]
#     for local in local_frequency: # Computing the statistics for each document.
#         doc_statistics = {}
#         for token in local:
#             if token in global_frequency:
#                 lf = local[token] # The local frequency
#                 gf = global_frequency[token] # The global frequency
#                 p = lf/gf
#                 entropy = (p * math.log2(p))/len(data)
#                 doc_statistics[token]=entropy
#         if len(doc_statistics):
#             log_entropy.append(doc_statistics)
#     return log_entropy

    # def get_log_entropy(line, term_frequency, doc_frequency):
    #     """
    #     """
    #     return True

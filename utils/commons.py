import pickle
import os
import operator

def get_data(path):
    f = open(path,'r',encoding='utf8')
    return f.read().split('\n')

def pickle_dict(dict, out_dir, file_name):
    if not out_dir.endswith('/'):
        out_dir+='/'
    with open(out_dir+file_name, 'wb') as out:
        pickle.dump(dict, out, protocol=pickle.HIGHEST_PROTOCOL)

def get_files_in_dir(dir):
    files = [f for f in os.listdir(dir)]
    return files

def list_splitter(a, n):
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))

def sort_dictionary(data,reverse=True):
    if reverse:
        return sorted(data.items(), key=operator.itemgetter(1), reverse=True)
    else:
        return sorted(data.items(), key=operator.itemgetter(1))

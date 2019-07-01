import pickle

def pickle_dict(dict, out_dir, file_name):
    if not out_dir.endswith('/'):
        out_dir+='/'
    with open(out_dir+file_name, 'wb') as out:
        pickle.dump(dict, out, protocol=pickle.HIGHEST_PROTOCOL)

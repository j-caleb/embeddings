"""
This is simply an alterative to starting the script at the terminal (which gets exhausting).
To use the config file settings simply exectute random_indexing.py with no arguments.
"""

config = {}

config['in_file']="F:\\github_projects\\data\\embeddings\\medline_sentences\\w2v_training_shuf.txt"
config['out_dir']="F:\\github_projects\\data\\embeddings\\medline_sentences\\models\\"
config['file_name']='ri_index'
config['seeds']=20
config['dim']=500
config['min_count']=10
config['print_status']=True
config['print_every']=500000
config['window_size']=None

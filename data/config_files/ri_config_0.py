"""
This is simply an alterative to starting the script at the terminal.
To use the config file settings simply exectute random_indexing.py with no arguments.
"""

config = {}

config['in_file']="F:\\github_projects\\data\\embeddings\\medline_sentences\\docs_test.json"
config['out_dir']="F:\\github_projects\\data\\embeddings\\medline_sentences\\models\\"
config['file_name']='ri_index'
config['seeds']=20
config['dim']=500
config['min_count']=25
config['print_status']=True
config['print_every']=10000
config['window_size']=None
config['sample']=None
config['mode']='ri'

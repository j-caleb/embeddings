"""
This is simply an alterative to starting the script at the terminal.
To use the config file settings simply exectute random_indexing.py with no arguments.
"""

config = {}

config['in_file']="F:\\github_projects\\data\\embeddings\\medline_sentences\\docs_small.json"
config['out_dir']="F:\\github_projects\\data\\embeddings\\medline_sentences\\ri_demo\\ri\\"
config['seeds']=20
config['dim']=500
config['min_count']=25
config['window_size']=5
config['sample']=None
config['mode']='ri'
config['text_field']='sent_tkns'
config['filename_field']='file_name'
config['metadata_field']='labels'
config['meta_min_count']=100

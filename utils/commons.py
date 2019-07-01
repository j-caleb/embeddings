def get_data(path):
    f = open(path,'r')
    return f.read().split('\n')

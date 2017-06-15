import os

def get_parent_dir_path(path):
    return os.path.split(os.path.split(path)[0])[0]
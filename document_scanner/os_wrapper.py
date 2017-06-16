import os


def get_parent_dir_path(path):
    return os.path.split(os.path.split(path)[0])[0]

BASE_DIR_PATH = get_parent_dir_path(os.path.realpath(__file__))
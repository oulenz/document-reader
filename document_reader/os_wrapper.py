import cv2
import numpy as np
import os

BASE_DIR_PATH = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
CURRENT_DIR_PATH = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
DEFAULT_PATH_DICT_PATH = os.path.join(BASE_DIR_PATH, 'data', 'paths.txt')

def list_images(folder_path):
    return [img_name for img_name in os.listdir(folder_path) if os.path.splitext(img_name)[1] in ['.jpg', '.JPG', '.png']]


def list_image_branches(folder_path):
    return [os.path.join(branch, img_name) for branch in list_subfolder_branches(folder_path) for img_name in list_images(os.path.join(folder_path, branch))]


def list_subfolders(folder_path):
    return [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]


def list_subfolder_branches(folder_path, branch=''):
    return sum([list_subfolder_branches(folder_path, os.path.join(branch, subfolder)) for subfolder in list_subfolders(os.path.join(folder_path, branch))], [branch])


def remove_dir(root):
    for filename in os.listdir(root):
        target = os.path.join(root, filename)
        if os.path.isdir(target):
            remove_dir(root=target)
        else:
            os.remove(target)

    os.rmdir(root)


def create_tmp_dir(root=os.path.join(CURRENT_DIR_PATH, 'tmp'), size=10, img_shape=(10, 10, 3)):
    os.mkdir(root)
    for label in ['x', 'y']:
        os.mkdir(os.path.join(root, label))
        for i in range(int(size / 2)):
            img = np.zeros(img_shape)
            path = os.path.join(root, label, str(i) + '.jpg')
            cv2.imwrite(path, img)

    return root
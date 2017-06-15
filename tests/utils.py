import os
import cv2
import numpy as np

curr_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))


def remove_dir(root):
    for filename in os.listdir(root):
        target = os.path.join(root, filename)
        if os.path.isdir(target):
            remove_dir(root=target)
        else:
            os.remove(target)

    os.rmdir(root)


def create_tmp_dir(root=os.path.join(curr_path, 'tmp'), size=10, img_shape=(10, 10, 3)):
    os.mkdir(root)
    for label in ['x', 'y']:
        os.mkdir(os.path.join(root, label))
        for i in range(int(size / 2)):
            img = np.zeros(img_shape)
            path = os.path.join(root, label, str(i) + '.jpg')
            cv2.imwrite(path, img)

    return root

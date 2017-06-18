import os
import shutil as su

import pandas as pd

import document_scanner as ds


def classify(model_path: str, num_classes: int, src_path: str, inception_graph_path: str):
    if os.path.isdir(src_path):
        img_paths = [os.path.join(src_path, x) for x in os.listdir(src_path) if
                     os.path.splitext(x)[1] in ['.jpg', '.png', '.JPG']]
    else:
        img_paths = [src_path]
    imgs = pd.DataFrame(img_paths, columns=['image_path'])
    imgs['class'] = ds.predict_with_model(inception_graph_path, model_path, num_classes, imgs['image_path'])

    folder_path = os.path.split(src_path)[0]
    for _, img in imgs.iterrows():
        category = str(img['class'])
        subfolder_path = os.path.join(folder_path, category)
        if not os.path.isdir(subfolder_path):
            os.mkdir(subfolder_path)
        su.copy(img['image_path'], subfolder_path)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Copy images into subfolders according to model prediction')
    parser.add_argument('model_path', help='The location of the model')
    parser.add_argument('num_classes', type=int, help='The number of classes of the model')
    parser.add_argument('src_path', help='The location of an image file or a folder of image files')
    parser.add_argument('inception_graph_path', help='The location of the inception graph', nargs='?')
    args = parser.parse_args()
    classify(args.model_path, args.num_classes, args.src_path, args.inception_graph_path)

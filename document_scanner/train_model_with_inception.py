import document_scanner as ds
import os
import tensorflow as tf

from tfwrapper import Dataset
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.utils.images import copy_image_folder
from tfwrapper.nets.pretrained import InceptionV3


def prepare_data(src, dest, flip_images):
    if not os.path.isdir(dest):
        os.mkdir(dest)

    for foldername in os.listdir(src):
        src_folder = os.path.join(src, foldername)

        if os.path.isdir(src_folder):
            dest_folder = os.path.join(dest, foldername)

            if not os.path.isdir(dest_folder):
                os.mkdir(dest_folder)

            copy_image_folder(src_folder, dest_folder, bw=True, h_flip=flip_images, v_flip=flip_images)


def train_model(config_path, src_path, model_name, flip_images):
    config = ds.parse_config(config_path)
    model_path = os.path.join(config['models_path'], model_name)
    features_path = os.path.join(src_path, model_name + '_features.csv')
    working_path = os.path.join(os.path.split(os.path.normpath(src_path))[0],
                                os.path.split(os.path.normpath(src_path))[1] + '_prepared')

    prepare_data(src_path, working_path, flip_images)
    number_of_classes = len([name for name in os.listdir(working_path) if not os.path.isfile(name)])

    inception = InceptionV3(graph_file=config['inception_graph_path'])
    features = inception.extract_features_from_datastructure(working_path, feature_file=features_path)

    dataset = Dataset(features=features)
    X, y, test_X, test_y, labels = dataset.getdata(balance=True, translate_labels=True, shuffle=True, onehot=True,
                                                   split=True)

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            nn = SingleLayerNeuralNet([X.shape[1]], number_of_classes, 1024, name=model_name, sess=sess, graph=graph)
            nn.train(X, y, epochs=10, sess=sess, verbose=True)
            nn.save(model_path, sess=sess)

    nn = SingleLayerNeuralNet([X.shape[1]], number_of_classes, 1024, name=model_name)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            nn.load(model_path, sess=sess)
            _, acc = nn.validate(X, y, sess=sess)
            print('Test accuracy: %d %%' % (acc * 100))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train model from images divided over subfolders')
    parser.add_argument('config_path', help='The config file')
    parser.add_argument('src_path', help='The folder containing the subfolders containing the images')
    parser.add_argument('model_name', help='The name to be used for the model')
    parser.add_argument('--dont_flip', dest='flip_images', action='store_false',
                        help='Prevents flipping of images for extra data')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Display pictures and values')
    args = parser.parse_args()
    train_model(args.config_path, args.src_path, args.model_name, args.flip_images)


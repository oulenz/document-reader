import cv2
import numpy as np
import os

from tfwrapper import ImageDataset
from tfwrapper.nets import CNN


def get_data(src, max_pixels = 10000):

    img_paths = []
    labels = []

    for label in [subfolder for subfolder in os.listdir(src) if os.path.isdir(os.path.join(src, subfolder))]:
        label_path = os.path.join(src, label)

        img_paths.extend([os.path.join(label_path, img_name) for img_name in os.listdir(label_path) if os.path.splitext(img_name)[1] == '.jpg'])
        labels.extend([label for img_name in os.listdir(label_path) if os.path.splitext(img_name)[1] == '.jpg'])

    first_img = cv2.imread(img_paths[0], cv2.COLOR_BGR2GRAY)
    height, width = first_img.shape[:2]
    pixels = height*width
    if pixels > max_pixels and max_pixels > 0:
        height = int(height/np.sqrt(pixels/max_pixels))
        width = int(width/np.sqrt(pixels/max_pixels))

    imgs = [cv2.resize(cv2.imread(img_path, cv2.COLOR_BGR2GRAY), (width, height)) for img_path in img_paths]

    X = np.array(imgs)
    Y = np.array(labels)

    return X, Y, (height, width, 1)


def train_custom_model(src_path, model_name):

    X, Y, (h, w, c) = get_data(src_path)
    dataset = ImageDataset(X=X, y=Y)
    dataset.preprocessor.blur(blur_steps=2, max_blur_sigma=2.5)
    dataset = dataset.normalize()
    dataset = dataset.balance()
    dataset = dataset.shuffle()
    dataset = dataset.translate_labels()
    dataset = dataset.onehot()
    train, test = dataset.split(0.8)
    train_X = np.reshape(train.X, [-1, h, w, c])
    train_y = train.y
    test_X = np.reshape(test.X, [-1, h, w, c])
    test_y = test.y
    num_classes = train_y.shape[1]

    name = model_name
    twice_reduce = lambda x: -1 * ((-1 * x) // 4)
    layers = [
        CNN.reshape([-1, h, w, c], name=name + '_reshape'),
        CNN.conv2d(filter=[5, 5], input_depth=1, depth=32, name=name + '_conv1'),
        CNN.maxpool2d(k=2, name=name + '_pool1'),
        CNN.conv2d(filter=[5, 5], input_depth=32, depth=64, name=name + '_conv2'),
        CNN.maxpool2d(k=2, name=name + '_pool2'),
        CNN.fullyconnected(input_size=twice_reduce(h) * twice_reduce(w) * 64, output_size=512,
                            name=name + '_fc'),
        CNN.out([512, num_classes], num_classes, name=name + '_pred')
    ]
    cnn = CNN([h, w, c], num_classes, layers, name=name)
    cnn.learning_rate = 0.01
    cnn.batch_size = 512
    cnn.train(train_X, train_y, epochs=5, verbose=True)
    _, acc = cnn.validate(test_X, test_y)
    print('Test accuracy: %.2f' % acc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train custum model from images divided over subfolders')
    parser.add_argument('src_path', help='The folder containing the subfolders containing the images')
    parser.add_argument('model_name', help='The name to be used for the model')
    args = parser.parse_args()
    train_custom_model(args.src_path, args.model_name)


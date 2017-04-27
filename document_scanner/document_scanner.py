# import the necessary packages
import cv2
import json
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from tfwrapper.nets import ShallowCNN
from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import InceptionV3
from typing import Callable


# canny edge method, not currently used
def find_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
    gray = cv2.GaussianBlur(gray, (17, 17), 0) # blur
    gray = cv2.bilateralFilter(gray,9,75,75) # what does this do again?
    edged = cv2.Canny(gray, 100, 200, apertureSize=5) # find edges

    # Blur, erode, dilate to remove noise. May have to look at net-effect
    edged = cv2.GaussianBlur(edged, (17, 17), 0)
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.erode(edged, kernel, iterations=1)
    edged = cv2.dilate(edged, kernel, iterations=1)
    return edged


def parse_config(path: str):
    config = {}
    with open(path, 'r') as config_file:
        for line in config_file.read().splitlines():
            if line.startswith('#'):
                continue # skip comments
            line = line.split('#', 1)[0] # strip comments
            k, v = line.split('=', 1)  # only consider first occurence of =
            config[k] = v

    return config


def get_keypoints_and_descriptors(img, orb, n):
    # Gets keypoints from all parts of the image by subdividing it into n*n parts and calculating keypoints separately
    keypoints = []
    descriptors = []
    h,w = img.shape[:2]
    for i in range(0,n):
        for j in range(0,n):
            mask = np.zeros((h, w), dtype = 'uint8')
            cv2.rectangle(mask, (i*w//n, j*h//n), ((i+1)*w//n, (j+1)*h//n), 255, cv2.FILLED)
            kp, des = orb.detectAndCompute(img, mask)
            if kp is not None:
                keypoints.extend(kp)
            # descriptors need to be in a numpy array
            if des is not None:
                descriptors.append(des)

    return keypoints, np.concatenate(descriptors)


def get_orb_values(template, photo, n):
    # Gathers into one method the detector-specific values
    # i.e. this method can be replaced by a corresponding method for a sift detector
    # n determines into how many rectangles (n*n) template and photo are subdivided
    # to force a more even distribution of keypoints

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=12000//(n*n),  # find many features
                         patchSize=31,  # granularity
                         edgeThreshold=1)  # edge margin to ignore

    # Find the keypoints and descriptors.
    kp_template, des_template = get_keypoints_and_descriptors(template, orb, n)
    kp_photo, des_photo = get_keypoints_and_descriptors(photo, orb, n)

    FLANN_INDEX_LSH = 6
    flann_index_params = dict(algorithm=FLANN_INDEX_LSH,
                              table_number=6,  # 6-12
                              key_size=12,  # 12-20
                              multi_probe_level=1)  # 1-2

    return kp_template, kp_photo, des_template, des_photo, flann_index_params


def resize(img, height):
    h, w = img.shape[:2]
    ratio = height/h
    return cv2.resize(img, (int(ratio*w), height))


def display(*imgs):
    for img in imgs:
        cv2.imshow('', resize(img, 650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_document_identification(template, photo, transform, mask, good, MIN_MATCH_COUNT, kp_template, kp_photo):
    photo_with_match = photo.copy()

    if len(good) > MIN_MATCH_COUNT:
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, transform)
        cv2.polylines(photo_with_match, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=mask.ravel().tolist() if mask is not None else None,  # draw only inliers
                       flags=2)

    photo_with_match = cv2.drawMatches(template, kp_template, photo_with_match, kp_photo, good, None, **draw_params)

    display(photo_with_match)


def get_matching_points(template, photo, debug: bool):
    kp_template, kp_photo, des_template, des_photo, flann_index_params = get_orb_values(template, photo, 3)

    flann_search_params = dict(checks=50)  # 50-100

    flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

    matches = flann.knnMatch(des_template, des_photo, k=2)

    # remove matches that are likely to be incorrect, using Lowe's ratio test
    good = [m for (m, n) in matches if m.distance < 0.7 * n.distance]  # 0.65-0.8, false negatives-false positives
    if debug:
        print(str(len(kp_template)) + ' points in template, ' + str(len(kp_photo)) + ' in photo, ' +
              str(len(matches)) + ' matches, ' + str(len(good)) + ' good matches')

    return kp_template, kp_photo, good


def find_transformation(template, photo, debug: bool):
    MIN_MATCH_COUNT = 10

    kp_template, kp_photo, matches = get_matching_points(template, photo, debug)

    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_photo[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # use RANSAC method to discount suspect matches
        transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #transform = cv2.estimateRigidTransform(src_pts, dst_pts, True)
    else:
        transform, mask = None, None

    if debug:
        show_document_identification(template, photo, transform, mask, matches, MIN_MATCH_COUNT, kp_template, kp_photo)

    return transform


def create_scan(template, photo, debug: bool):
    transform = find_transformation(template.copy(), photo.copy(), debug)

    # inverse the found transformation to retrieve the document
    h, w = template.shape
    scan = cv2.warpPerspective(photo, np.linalg.inv(transform), (w, h))

    # show the original and scanned images
    if debug:
        display(resize(photo, 650), resize(scan, 650))

    return scan


def find_document(template, photo, debug: bool):
    h, w = template.shape
    h_scan = int(h * 1.2)
    small = resize(photo, h_scan)

    return create_scan(template, small, debug)


def parse_boxes(config):

    boxes = pd.read_csv(config['boxes_path'], delimiter='|', comment='#')
    boxes['coords'] = boxes['coords'].apply(lambda x: tuple([int(y) for y in x.split(':')]))  # (l, r, u, d)
    box_types = pd.read_csv(config['box_types_path'], delimiter='|', comment='#', dtype={'uses_inception': bool})
    boxes = boxes.merge(box_types, left_on='type', right_on='type')

    if False:
        areas_file = open(config['boxes_path'], 'r')

        coord_dict = {}
        type_dict = {}
        for line in areas_file.readlines():
            tokens = line.split(',')
            name = tokens[0]
            type_dict[name] = tokens[1]
            l = int(tokens[2].split(':')[0])
            u = int(tokens[2].split(':')[1])
            r = int(tokens[3].split(':')[0])
            d = int(tokens[3].split(':')[1])
            coord_dict[name] = (l, r, u, d)

    required_columns = {'name', 'coords', 'type', 'model_path', 'uses_inception', 'num_classes', 'crop'}

    for column in required_columns - set(boxes.columns):
         print('Error: required column ' + column + ' missing in data files')

    boxes = boxes.set_index('name')

    return boxes


def show_boxes(photo, boxes):
    import random
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(16, 13))
    plt.imshow(photo)

    colours = [(1, 1, 1)] + [(random.random(), random.random(), random.random()) for i in range(255)]
    new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colours, N=256)
    for name, (l, r, u, d) in boxes['coords'].iteritems():
        ul = (l, u)
        w = r - l
        h = d - u
        colour = colours[random.randint(0, 255)]
        rect = patches.Rectangle(ul, w, h, linewidth=1, alpha=0.5, color=colour, label=name)
        plt.gca().add_patch(rect)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, labels, fontsize=8, ncol=2, bbox_to_anchor=(-0.6, 1), loc='upper left')

    plt.show()
    return


def pad_coords(coords, scan_shape):
    padding = 8
    l, r, u, d = coords
    d_max, r_max = scan_shape

    return (max(l - padding, 0),
           min(r + padding, r_max),
           max(u - padding, 0),
           min(d + padding, d_max))


def crop_box(scan, coords):
    l, r, u, d = pad_coords(coords, scan.shape)
    return scan[u:d, l:r]


def crop_boxes(scan, boxes):
    box_selection = boxes[boxes['crop'] == 1].copy()
    box_selection['crop'] = box_selection['coords'].apply(lambda x: crop_box(scan, x))
    return box_selection


def predict_with_inception(inception_graph_path: str, model_path: str, model_name: str, num_labels: int, crops):
    with tf.Session() as sess:
        inception = InceptionV3(graph_file=inception_graph_path, sess=sess)
        img_features = inception.extract_features_from_imgs(crops, sess=sess)
        nn = SingleLayerNeuralNet([len(img_features[0])], num_labels, 1024, name=model_name, sess=sess)
        nn.load(model_path, sess=sess)
        yhat = pd.Series(nn.predict(np.array(img_features.tolist()), sess=sess).tolist(), index=crops.index)
    tf.reset_default_graph()
    return yhat


# TODO: handle non-shallow cnns
def predict_with_cnn(model_path: str, model_name: str, c: int, num_labels: int, crops):
    img_features = np.array(crops.tolist())
    n, h, w = img_features.shape[:3]
    img_features = np.reshape(img_features, [n, h, w, c])

    with tf.Session() as sess:
        cnn = ShallowCNN([h, w, c], num_labels, name=model_name, sess=sess)
        cnn.load(model_path, sess=sess)
        yhat = pd.Series(cnn.predict(img_features, sess=sess).tolist(), index=crops.index)
    tf.reset_default_graph()
    return yhat


def predict_with_model(inception_graph_path: str, model_path: str, model_config, crops):
    model_name = model_config['name']
    num_labels = model_config['y_size']

    h, w, c = model_config['X_shape']
    crops = crops.apply(lambda x: cv2.resize(x, (w, h)))
    if c == 3:
        crops = crops.apply(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB))

    yhat = (predict_with_inception(inception_graph_path, model_path, model_name, num_labels, crops)
                    if inception_graph_path is not None
                    else predict_with_cnn(model_path, model_name, c, num_labels, crops))
    return yhat.apply(lambda x: np.argmax(x))


def classify_boxes(boxes, config, debug):
    values = {}

    for (model_path, uses_inception, num_classes), model_boxes in boxes.groupby(['model_path', 'uses_inception', 'num_classes']):
        inception_graph_path = config['inception_graph_path'] if uses_inception else None
        with open(model_path + '.tw') as f:
            model_config = json.load(f)
        classes = predict_with_model(inception_graph_path, model_path, model_config, model_boxes['crop'])
        labels = classes.apply(lambda x: model_config['labels'][x])
        values.update(labels.to_dict())

    if debug:
        print(values)

    return values


def read_document(scan, config, debug: bool):
    boxes = parse_boxes(config)
    if debug:
        show_boxes(scan, boxes)
    boxes = crop_boxes(scan, boxes)
    values = classify_boxes(boxes, config, debug)
    return values


def scan_document(config_path: str, img_path: str, debug: bool):
    config = parse_config(config_path)
    if debug:
        print(config)
    template = cv2.imread(config['template_path'], 0)
    photo = cv2.imread(img_path, 0)
    scan = find_document(template, photo, debug)
    values = read_document(scan, config, debug)
    return values



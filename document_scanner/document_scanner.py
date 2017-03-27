# import the necessary packages
import os
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import InceptionV3


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


def parse_config(path: str, image_path: str = None):
    config = {}
    with open(path, 'r') as config_file:
        for line in config_file.read().splitlines():
            if line.startswith('#'):
                continue # skip comments
            line = line.split('#', 1)[0] # strip comments
            k, v = line.split('=', 1)  # only consider first occurence of =
            config[k] = v

    if image_path is not None:
        config['image_path'] = image_path

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
            keypoints.extend(kp)
            # descriptors need to be in a numpy array
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


def find_transformation(template, photo, debug: bool):
    MIN_MATCH_COUNT = 10

    kp_template, kp_photo, des_template, des_photo, flann_index_params = get_orb_values(template, photo, 3)

    flann_search_params = dict(checks=50)  # 50-100

    flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

    matches = flann.knnMatch(des_template, des_photo, k=2)

    # remove matches that are likely to be incorrect, using Lowe's ratio test
    good = [m for (m, n) in matches if m.distance < 0.7 * n.distance]  # 0.65-0.8, false negatives-false positives
    if debug:
        print(str(len(kp_template)) + ' points in template, ' + str(len(kp_photo)) + ' in photo, ' +
              str(len(matches)) + ' matches, ' + str(len(good)) + ' good matches')

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_photo[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # use RANSAC method to discount suspect matches
        transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #transform = cv2.estimateRigidTransform(src_pts, dst_pts, True)
    else:
        transform, mask = None, None

    if debug:
        show_document_identification(template, photo, transform, mask, good, MIN_MATCH_COUNT, kp_template, kp_photo)

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


def find_document(template, image_path, debug: bool):
    photo = cv2.imread(image_path, 0)
    h, w = template.shape
    h_scan = int(h * 1.2)
    small = resize(photo, h_scan)

    return create_scan(template, small, debug)


def parse_boxes(config):

    boxes = pd.read_csv(config['boxes_path'], delimiter='|', comment='#')
    boxes['coords'] = boxes['coords'].apply(lambda x: tuple([int(y) for y in x.split(':')]))  # (l, r, u, d)
    box_types = pd.read_csv(config['box_types_path'], delimiter='|', comment='#')
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

    required_columns = {'name', 'coords', 'type', 'model_path', 'num_classes', 'crop'}

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
    r_max, d_max = scan_shape

    return (max(l - padding, 0),
           min(r + padding, r_max),
           max(u - padding, 0),
           min(d + padding, d_max))


def crop_boxes(scan, boxes, img_path, config) -> None:
    box_selection = boxes[boxes['crop'] == 1]

    folder_path, file = os.path.split(img_path)
    file_stem, file_ext = os.path.splitext(file)
    output_path = config['output_path']
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    for box_name, coords in box_selection['coords'].iteritems():
        l, r, u, d = pad_coords(coords, scan.shape)
        crop = scan[u:d, l:r]
        filename = os.path.join(output_path, file_stem + '_' + box_name + file_ext)
        print(filename)
        cv2.imwrite(filename, crop)
        box_selection.loc[box_name,'crop_path'] = filename

    return box_selection


def classify_boxes(boxes, config, debug, scan):
    inception = InceptionV3(graph_file=config['inception_graph_path'])
    boxes['features'] = inception.extract_features_from_files(boxes['crop_path'].tolist()).tolist()

    for (model_name, model_path, num_classes), model_boxes in boxes.groupby(['type', 'model_path', 'num_classes']):
        nn = SingleLayerNeuralNet([len(model_boxes.ix[0, 'features'])], num_classes, 1024, name=model_name)
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                nn.load(model_path, sess=sess)
                yhat = nn.predict(np.array(model_boxes['features'].tolist()), sess=sess)
                print(yhat)
                model_boxes['yhat'] = yhat.tolist()

        if debug:
            for name, box in model_boxes.iterrows():
                category = str(np.argmax(box['yhat']))
                print(name + ' ' + category)
                l,r,u,d = box['coords']
                # TODO: needs tweaking
                cv2.putText(scan, category, org=(l, d), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0))

            display(scan)

    return


def read_document(scan, img_path, config, debug: bool):
    boxes = parse_boxes(config)
    if debug:
        show_boxes(scan, boxes)
    boxes = crop_boxes(scan, boxes, img_path, config)
    classify_boxes(boxes, config, debug, scan)
    return


def scan_document(config_path: str, image_path: str, debug: bool) -> None:
    config = parse_config(config_path, image_path)
    if debug:
        print(config)
    template = cv2.imread(config['template_path'], 0)
    scan = find_document(template, config['image_path'], debug)
    read_document(scan, config['image_path'], config, debug)
    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Read out form from photo')
    parser.add_argument('config_path', help='The location of the config file')
    parser.add_argument('image_path', help='The location of the image file (overrides path in config file)', nargs='?')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Display pictures and values')
    args = parser.parse_args()
    scan_document(args.config_path, args.image_path, args.debug)



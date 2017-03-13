# import the necessary packages
import os
from typing import Callable

import cv2
import imutils
import numpy as np
import tensorflow as tf

from tfwrapper.nets import SingleLayerNeuralNet
from tfwrapper.nets.pretrained import InceptionV3



def get_config(path: str, image_path: str):
    config_file = open(path, 'r')

    config = {}
    for line in config_file.read().splitlines():
        k, v = line.split('=', 1)  # only consider first occurence of =
        config[k] = v

    if image_path is not None:
        config['image_path'] = image_path

    return config


def get_orb_values(template, photo):
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=10000,  # find many features
                         patchSize=31,  # granularity
                         edgeThreshold=0)  # edge margin to ignore

    # find the keypoints and descriptors
    kp_template, des_template = orb.detectAndCompute(template, None)
    kp_photo, des_photo = orb.detectAndCompute(photo, None)

    FLANN_INDEX_LSH = 6
    flann_index_params = dict(algorithm=FLANN_INDEX_LSH,
                              table_number=6,  # 6-12
                              key_size=12,  # 12-20
                              multi_probe_level=1)  # 1-2

    return kp_template, kp_photo, des_template, des_photo, flann_index_params


def find_transformation(template, photo, debug: bool):
    MIN_MATCH_COUNT = 10

    # kp_template, kp_photo, des_template, des_photo, index_params = get_sift_values(template, photo)
    kp_template, kp_photo, des_template, des_photo, flann_index_params = get_orb_values(template, photo)

    flann_search_params = dict(checks=50)  # 50-100

    flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # bf = cv2.BFMatcher()

    matches = flann.knnMatch(des_template, des_photo, k=2)
    # matches = bf.knnMatch(des_template, des_photo, k=2)


    # remove matches that are likely to be incorrect, using Lowe's ratio test
    # good = []
    # for match in matches:
    #    if len(match) == 2:
    #        m, n = match
    #        if m.distance < 0.7 * n.distance:  # 0.65-0.8, false negatives-false positives
    #            good.append(m)
    good = [m for (m, n) in matches if m.distance < 0.7 * n.distance]  # 0.65-0.8, false negatives-false positives
    if debug:
        print(str(len(kp_template)) + ' matches in template, ' + str(len(kp_photo)) + ' in photo, ' +
              str(len(matches)) + ' matches, ' + str(len(good)) + ' good matches')

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_photo[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        transform = None

    if debug:
        if len(good) > MIN_MATCH_COUNT:
            matchesMask = mask.ravel().tolist()

            h, w = template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, transform)

            photo_with_lines = photo.copy()
            cv2.polylines(photo_with_lines, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
            photo_with_lines = photo

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        photo_with_match = cv2.drawMatches(template, kp_template, photo_with_lines, kp_photo, good, None, **draw_params)

        # plt.imshow(img3, 'gray'), plt.show()
        cv2.imshow("Match", imutils.resize(photo_with_match, height=650))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return transform


def create_scan(template, photo, debug: bool):
    h, w = template.shape
    transform = find_transformation(template, photo, debug)

    # inverse the found transformation to retrieve the document
    scan = cv2.warpPerspective(photo, np.linalg.inv(transform), (w, h))

    # show the original and scanned images
    if debug:
        cv2.imshow("Original", imutils.resize(photo, height=650))
        cv2.imshow("Scanned", imutils.resize(scan, height=650))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return scan


def find_document(template, config, debug: bool):
    photo = cv2.imread(config['image_path'], 0)
    h, w = template.shape
    h_scan = int(h * 1.2)
    small = imutils.resize(photo, height=h_scan)
    scan = create_scan(template, small, debug)

    return scan


def parse_coords(config):
    areas_file = open(config['boxes_path'], 'r')

    coord_dict = {}
    for line in areas_file.readlines():
        tokens = line.split(',')
        name = tokens[0]
        l = int(tokens[1].split(':')[0])
        u = int(tokens[1].split(':')[1])
        r = int(tokens[2].split(':')[0])
        d = int(tokens[2].split(':')[1])
        coord_dict[name] = (l, r, u, d)

    return coord_dict


def get_sections(path: str):
    sections_file = open(path, 'r')

    sections = []
    for line in sections_file.read().splitlines():
        section = {}
        tokens = line.split(':', 2)
        section['name'] = tokens[0]
        try:
            section['coords'] = tuple(int(x) for x in tokens[1].split(','))
        except Exception as e:
            print('Error: non-number among dimensions ' + tokens[1] + ' of ' + tokens[0])
        section['elements'] = tokens[2].split(',')

        sections.append(section)

    return sections


def find_section(photo, section_template, debug: bool):
    return create_scan(section_template, photo, debug)


def is_out_of_bounds(coords, bounds):
    l, r, u, d = coords
    l_min, r_max, u_min, d_max = bounds
    return min(l - l_min, r_max - r, u - u_min, d_max - d) < 0


def shift_coords(coords, hor_shift, vert_shift):
    l, r, u, d = coords
    return (l - hor_shift, r - hor_shift, u - vert_shift, d - vert_shift)


def subselect_section(coord_dict, section):
    elements = section['elements']

    subselect = {}

    for k in elements:
        if not k in coord_dict:
            print('Warning: element ' + k + ' not in coordinate file')
            continue
        if is_out_of_bounds(coord_dict[k], section['coords']):
            print('Warning: coordinates ' + str(coord_dict[k]) + ' of element ' + k +
                  'not contained within coordinates ' + str(section['coords']) + ' of section ')
            continue
        l_section, _, u_section, _ = section['coords']
        subselect[k] = shift_coords(coord_dict[k], l_section, u_section)

    return subselect


def show_boxes(photo, coord_dict):
    import random
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(16, 13))
    plt.imshow(photo)

    colours = [(1, 1, 1)] + [(random.random(), random.random(), random.random()) for i in range(255)]
    new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colours, N=256)
    for name, (l, r, u, d) in coord_dict.items():
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


def get_scale_factors(section, scan):
    # Return values differ from 1 only if the recovered scan was not resized to the dimensions of the template
    l, r, u, d = section['coords']
    h_scan, w_scan = scan.shape
    return w_scan / (r - l), h_scan / (d - u)


def scale_and_pad_coords(coords, scale_factors, padding, r_max, d_max):
    l, r, u, d = coords
    vert_scale, hor_scale = scale_factors

    return (max(int(l * hor_scale) - padding, 0),
           min(int(r * hor_scale) + padding, r_max),
           max(int(u * vert_scale) - padding, 0),
           min(int(d * vert_scale) + padding, d_max))


def crop_boxes(scan, config, coord_dict, scale_factors) -> None:
    padding = 8
    output_path = config['output_path']

    folder_path, file = os.path.split(config['image_path'])
    file_stem, file_ext = os.path.splitext(file)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    h_scan, w_scan = scan.shape

    filenames = []
    for element, coords in coord_dict.items():
        l, r, u, d = scale_and_pad_coords(coords, scale_factors, padding, w_scan, h_scan)
        crop = scan[u:d, l:r]
        filename = os.path.join(output_path, file_stem + '_' + element + file_ext)
        cv2.imwrite(filename, crop)
        filenames.append(filename)

    print('Found ' + str(len(filenames)) + ' boxes')
    return filenames


def find_boxes(scan, coord_dict, section, config, debug: bool) -> None:
    coord_dict = subselect_section(coord_dict, section)
    if debug:
        show_boxes(scan, coord_dict)
    scale_factors = get_scale_factors(section, scan)
    return crop_boxes(scan, config, coord_dict, scale_factors)


def read_document(scan, template, config, debug: bool) -> None:
    coord_dict = parse_coords(config)
    sections = get_sections(config['sections_path'])
    filenames = []

    for section in sections:
        l, r, u, d = section['coords']
        section_template = template[u:d, l:r]
        scan = find_section(scan, section_template, debug)
        filenames += find_boxes(scan, coord_dict, section, config, debug)
    return filenames


def scan_document(config_path: str, image_path: str, debug: bool) -> None:
    config = get_config(config_path, image_path)
    if debug:
        print(config)
    template = cv2.imread(config['template_path'], 0)
    scan = find_document(template, config, debug)
    file_names = read_document(scan, template, config, debug)
    classify_boxes(file_names, config)
    return

def col_nr(val):
    if val == 'A':
        return 0
    else:
        return 1

def img_repr(val):
    img = np.ones((100, 100, 3)) * 255
    #cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) 
    if val is None:
        img = img
    elif val < 0.5:
        cv2.line(img, (50, 90), (90, 10), (0, 255, 0), 3)
        cv2.line(img, (25, 50), (50, 90), (0, 255, 0), 3)
    else:
        cv2.line(img, (10, 10), (90, 90), (0, 0, 255), 3)
        cv2.line(img, (10, 90), (90, 10), (0, 0, 255), 3)

    return img

def str_repr(val):
    if val == 0:
        return '[X]'
    else:
        return '[ ]'

def classify_boxes(filenames, config):
    #inception = InceptionV3(graph_file='/Users/esten/ml/imagenet/classify_image_graph_def.pb')
    inception = InceptionV3(graph_file=config['inception_graph_path'])
    features = inception.extract_features_from_files(filenames)

    nn = SingleLayerNeuralNet(features.shape[1], 2, 1024, name='Checkboxes')
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            nn.load(config['checkbox_model_path'], sess=sess)
            yhat = nn.predict(features, sess=sess)
    
    grid = np.ones([int(len(filenames) / 2) - 1, 2])
    for i in range(len(filenames)):
        filename = filenames[i]
        tokens = filename.split('_')

        try:
            row = int(tokens[-2][1:]) - 1
            col = col_nr(tokens[-1].split('.')[0])
            grid[row][col] = np.argmax(yhat[i])
            if config['debug']:
                img = cv2.imread(filename)
                img = cv2.resize(img, (100, 100))
                rep = img_repr(grid[row][col])
                cv2.imshow('Cell', img)
                cv2.imshow('Class', rep)
                cv2.moveWindow('Cell', 700, 400)
                cv2.moveWindow('Class', 700, 520)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception:
            continue

    for i in range(len(grid)):
        print(str_repr(grid[i][0]) + '\t' + str_repr(grid[i][1]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Read out form from photo')
    parser.add_argument('config_path', help='The location of the config file')
    parser.add_argument('image_path', help='The location of the image file (overrides path in config file)', nargs='?')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Display pictures and values')
    args = parser.parse_args()
    scan_document(args.config_path, args.image_path, args.debug)

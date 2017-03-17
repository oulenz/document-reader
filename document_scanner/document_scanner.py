# import the necessary packages
import os
from typing import Callable

import cv2
import numpy as np
import pandas as pd
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


def resize(img, height):
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    ratio = height/h
    return cv2.resize(img, (int(ratio*w), height))


def find_transformation(template, photo, debug: bool):
    print('test')
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
        print(str(len(kp_template)) + ' points in template, ' + str(len(kp_photo)) + ' in photo, ' +
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
        cv2.imshow("Match", resize(photo_with_match, 650))
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
        cv2.imshow("Original", resize(photo, 650))
        cv2.imshow("Scanned", resize(scan, 650))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return scan


def find_document(template, config, debug: bool):
    photo = cv2.imread(config['image_path'], 0)
    h, w = template.shape
    h_scan = int(h * 1.2)
    small = resize(photo, h_scan)
    scan = create_scan(template, small, debug)

    return scan


def parse_coords(config):

    boxes = pd.read_csv(config['boxes_path'], delimiter='|')
    boxes['coords'] = boxes['coords'].apply(lambda x: tuple([int(y) for y in x.split(':')]))  # (l, r, u, d)
    box_types = pd.read_csv(config['box_types_path'], delimiter='|')
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

    required_columns = {'name', 'coords', 'type', 'model_path'}

    for column in required_columns - set(boxes.columns):
         print('Error: required column ' + column + ' missing in data files')

    boxes = boxes.set_index('name')

    return boxes


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


def subselect_section(boxes, section):
    elements = section['elements']

    for box_name in elements:
        if box_name not in boxes.index:
            print('Warning: box ' + box_name + ' not in box file')
        elif is_out_of_bounds(boxes.loc[box_name, 'coords'], section['coords']):
            print('Warning: coordinates ' + str(boxes.loc[box_name, 'coords']) + ' of box ' + box_name +
                  'not contained within coordinates ' + str(section['coords']) + ' of section ')

    l_section, _, u_section, _ = section['coords']

    box_selection = boxes.copy()[boxes.index.isin(elements)]
    box_selection['section_coords'] = box_selection['coords'].apply(lambda x: shift_coords(x, l_section, u_section))

    return box_selection


def show_boxes(photo, box_selection):
    import random
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(16, 13))
    plt.imshow(photo)

    colours = [(1, 1, 1)] + [(random.random(), random.random(), random.random()) for i in range(255)]
    new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colours, N=256)
    for name, (l, r, u, d) in box_selection['section_coords'].iteritems():
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


def crop_boxes(scan, config, box_selection, scale_factors) -> None:
    padding = 8
    output_path = config['output_path']

    folder_path, file = os.path.split(config['image_path'])
    file_stem, file_ext = os.path.splitext(file)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    h_scan, w_scan = scan.shape

    for box_name, coords in box_selection['section_coords'].iteritems():
        l, r, u, d = scale_and_pad_coords(coords, scale_factors, padding, w_scan, h_scan)
        crop = scan[u:d, l:r]
        filename = os.path.join(output_path, file_stem + '_' + box_name + file_ext)
        cv2.imwrite(filename, crop)
        box_selection.loc[box_name,'crop_path'] = filename

    return box_selection


def find_boxes(scan, boxes, section, config, debug: bool):
    box_selection = subselect_section(boxes, section)
    if debug:
        show_boxes(scan, box_selection)
    scale_factors = get_scale_factors(section, scan)
    return crop_boxes(scan, config, box_selection, scale_factors)


def read_document(scan, template, config, debug: bool):
    boxes = parse_coords(config)
    sections = get_sections(config['sections_path'])
    box_selections = []

    for section in sections:
        l, r, u, d = section['coords']
        section_template = template[u:d, l:r]
        scan = find_section(scan, section_template, debug)
        box_selections.append(find_boxes(scan, boxes, section, config, debug))
    return pd.concat(box_selections)


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


def classify_boxes(boxes, config, debug, scan):
    inception = InceptionV3(graph_file=config['inception_graph_path'])
    #features = inception.extract_features_from_files(filenames)
    boxes['features'] = inception.extract_features_from_files(boxes['crop_path'].tolist()).tolist()

    for model_path, model_boxes in boxes.groupby('model_path'):
        nn = SingleLayerNeuralNet([len(model_boxes['features'])], 2, 1024, name='Checkboxes')
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                nn.load(model_path, sess=sess)
                yhat = nn.predict(np.array(model_boxes['features'].tolist()), sess=sess)
                print(yhat)
                model_boxes['yhat'] = yhat.tolist()

        for name, box in model_boxes.iterrows():
            print(name + ' ' + str(np.argmax(box['yhat'])))

        if debug:

            for name, box in model_boxes.iterrows():
                l,r,u,d = box['coords']
                cv2.putText(scan, str(np.argmax(box['yhat'])), org=(l, d), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0))

            cv2.imshow('Detected', resize(scan, 650))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return


def scan_document(config_path: str, image_path: str, debug: bool) -> None:
    config = get_config(config_path, image_path)
    if debug:
        print(config)
    template = cv2.imread(config['template_path'], 0)
    scan = find_document(template, config, debug)
    boxes = read_document(scan, template, config, debug)
    classify_boxes(boxes, config, debug, scan)
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


def characters(img):
    h,w = img.shape


    return
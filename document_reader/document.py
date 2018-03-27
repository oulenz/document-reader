import cv2
import json
import numpy as np
import os

from abc import ABC
from typing import Any, Dict

from document_reader.cv_wrapper import display, find_transform_and_mask, get_keypoints_and_descriptors, get_matching_points, resize, reverse_transformation
from document_reader.py_wrapper import coalesce, store_time


class Image_data(ABC):
    
    def __init__(self):
        self.keypoints = None
        self.kp_descriptors = None
        self.photo = None
        
    @classmethod
    def of_photo(cls, photo, orb):
        image_data = cls()
        image_data.photo = photo
        image_data.keypoints, image_data.kp_descriptors = get_keypoints_and_descriptors(photo, orb)
        return image_data


class DocumentScan(ABC):

    MIN_MATCH_COUNT = 10

    def __init__(self):
        self._method_times = []
        self.error_reason = None
        self.mask = None
        self.matches = None
        self.original = None
        self.target_data = Image_data()
        self.result = None
        self.template_data = Image_data()
        self.transform = None

    @classmethod
    def from_photo(cls, photo, template_data, orb, proxy=None):
        scan = cls()
        scan.original = photo
        scan.template_data = template_data
        proxy = cls._resize_to_template(coalesce(proxy, photo), template_data.photo.shape)
        scan.target_data = Image_data.of_photo(proxy, orb)
        scan.matches = scan._find_matches(scan.target_data, template_data)
        if not scan._can_create_scan():
            return scan
        scan.transform, scan.mask = find_transform_and_mask(template_data.keypoints, scan.target_data.keypoints, scan.matches)
        scan.result = scan._create_scan(photo, template_data, scan.transform)
        return scan

    @store_time
    def _find_matches(self, photo_data, template_data):
        return get_matching_points(template_data.kp_descriptors, photo_data.kp_descriptors) if photo_data.kp_descriptors is not None else None

    def _can_create_scan(self):
        return self.matches and len(self.matches) > self.MIN_MATCH_COUNT

    @store_time
    def _create_scan(self, photo, template_data, transform):
        template_shape = template_data.photo.shape
        resized = self._resize_to_template(photo, template_shape)
        return reverse_transformation(resized, transform, template_shape)

    @staticmethod
    def _resize_to_template(photo, template_shape):
        multiplication_factor = 1.2
        h, w = template_shape[:2]
        length_to_use = int(max(h, w) * multiplication_factor)
        return resize(photo, length_to_use)

    def show_match_with_template(self):
        photo_with_match = self.target_data.photo.copy()

        if self._can_create_scan():
            h, w = self.template_data.photo.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, self.transform)
            cv2.polylines(photo_with_match, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - %d/%d" % (len(self.matches), self.MIN_MATCH_COUNT))

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=self.mask.ravel().tolist() if self.mask is not None else None,  # draw only inliers
                           flags=2)

        photo_with_match = cv2.drawMatches(self.template_data.photo, self.template_data.keypoints, photo_with_match, self.target_data.keypoints, self.matches, None, **draw_params)

        display(photo_with_match)
        return

    def print_template_match_quality(self):
        print(str(len(self.template_data.keypoints)) + ' points in template, ' + str(len(self.target_data.keypoints)) + ' in photo, ' +
              str(len(self.matches)) + ' good matches')
        return

    def show_original_and_result(self):
        # show the original and scanned images
        display(resize(self.original, 650), resize(self.result, 650))

    @store_time
    def get_corners(self):
        # Returns the coordinates of the corners of the scan in the original image
        h, w = self.template_data.photo.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        corners = np.int32(cv2.perspectiveTransform(pts, self.transform))
        return corners


class Document(ABC):

    def __init__(self):
        self._method_times = []
        self.document_type_name = None
        self.field_df = None
        self.photo = None
        self.scan = DocumentScan()
        self.scan_retries = None

    @classmethod
    def from_parts(cls, photo, document_type_name, scan: DocumentScan, field_df, scan_retries=None):
        document = cls()
        document.photo = photo
        document.document_type_name = document_type_name
        document.scan = scan
        document.scan_retries = scan_retries
        document.field_df = field_df
        return document

    def get_field_labels(self):
        return self.field_df['label'] if self.field_df is not None else None

    def get_field_labels_json(self):
        return self.field_df['label'].to_json() if self.field_df is not None else None

    def get_field_labels_dict(self):
        return self.field_df['label'].to_dict() if self.field_df is not None else None

    def show_boxes(self, field_data_df):
        import random
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig = plt.figure(figsize=(16, 13))
        plt.imshow(self.scan.result)

        colours = [(1, 1, 1)] + [(random.random(), random.random(), random.random()) for i in range(255)]
        new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colours, N=256)
        for name, (l, r, u, d) in field_data_df['lrud'].items():
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

    def get_case_log(self, **case_log) -> Dict[str, Any]:
        case_log['predictions'] = self.get_prediction_dict()
        case_log['method_times'] = self.scan._method_times.extend(self._method_times)
        return case_log
    
    def save_images_and_case_log(self, log_path, case_id):
        case_log = self.get_case_log()
        case_log['case_id'] = case_id
        case_log['log_path'] = log_path
        
        case_log['image_paths'] = {}
        imgs_to_save = {}
        imgs_to_save['photo'] = self.photo
        if self.scan is not None:
            imgs_to_save['photo_grey'] = self.scan.original
            imgs_to_save['photo_proxy'] = self.scan.target_data.photo
            imgs_to_save['scan'] = self.scan.result
        for name, img in imgs_to_save.items():
            img_output_path = os.path.join(log_path, '{}_{}.jpg'.format(case_id, name))
            case_log['image_paths'][name] = img_output_path
            cv2.imwrite(img_output_path, img)
        if getattr(self, 'field_df', None) is not None:
            case_log['image_paths']['crops'] = {}
            for field_name, row in self.field_df.iterrows():
                crop_output_path = os.path.join(log_path, '{}_crop_{}.jpg'.format(case_id, field_name))
                case_log['image_paths']['crops'][field_name] = crop_output_path
                cv2.imwrite(crop_output_path, row['crop'])
        
        case_log_output_path = os.path.join(log_path, '{}_case_log.json'.format(case_id))
        with open(case_log_output_path, 'w') as fp:
            json.dump(case_log, fp)
        return None
    
    def get_prediction_dict(self):
        prediction_dict = {}
        prediction_dict['document_type_name'] = getattr(self, 'document_type_name')
        prediction_dict['field_labels'] = self.get_field_labels_dict()
        return prediction_dict
    
    def show_debug_information(self, field_data_df):
        self.scan.print_template_match_quality()
        self.scan.show_match_with_template()
        if self.scan is not None:
            return
        self.scan.show_original_and_result()
        self.show_boxes(field_data_df)
        print(self.get_field_labels_json())
    
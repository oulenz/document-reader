import cv2
import json
import numpy as np
import os

from abc import ABC

from document_scanner.cv_wrapper import crop_sections, display, find_transformation_and_mask, get_keypoints_and_descriptors, get_matching_points, resize, reverse_transformation, sharpen_image
import document_scanner.tfw_wrapper as tfw_wrapper
from document_scanner.py_wrapper import store_time

MIN_MATCH_COUNT = 10


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


class Document(ABC):

    def __init__(self):
        self._method_times = []
        self.logic = None
        self.document_type_name = None
        self.error_reason = None
        self.field_df = None
        self.image_data = None
        self.img_path = None
        self.mask = None
        self.matches = None
        self.photo = None
        self.photo_grey = None
        self.scan = None
        self.template_data = None
        self.transform = None
        
    @classmethod
    def from_path(cls, img_path:str, business_logic_class):
        document = cls()
        document.img_path = img_path
        document.photo = cv2.imread(img_path, 1)
        document.photo_grey = cv2.imread(img_path, 0)
        document.logic = business_logic_class()
        return document

    @store_time
    def predict_document_type(self, model_and_labels, pretrained_client=None, mock_document_type_name=None):
        self.document_type_name = mock_document_type_name or tfw_wrapper.label_img(self.photo_grey, *model_and_labels, pretrained_client)
        return

    @store_time
    def find_match(self, template, orb):
        self.template_data = template
        photo_grey_to_use = cv2.cvtColor(sharpen_image(self.photo), cv2.COLOR_RGB2GRAY)
        resized_to_use = self.resize_to_template(photo_grey_to_use, template.photo.shape)
        self.image_data = Image_data.of_photo(resized_to_use, orb)
        self.matches = get_matching_points(template.kp_descriptors, self.image_data.kp_descriptors) if self.image_data.kp_descriptors is not None else None
        return
    
    @staticmethod
    def resize_to_template(photo, template_shape):
        multiplication_factor = 1.2
        h, w = template_shape[:2]
        length_to_use = int(max(h, w) * multiplication_factor)
        return resize(photo, length_to_use)

    def can_create_scan(self):
        return self.matches and len(self.matches) > MIN_MATCH_COUNT

    @store_time
    def find_transform_and_mask(self):
        self.transform, self.mask = find_transformation_and_mask(self.template_data.keypoints, self.image_data.keypoints, self.matches)
        return

    @store_time
    def create_scan(self):
        template_shape = self.template_data.photo.shape
        resized = self.resize_to_template(self.photo_grey, template_shape)
        self.scan = reverse_transformation(resized, self.transform, template_shape)
        return

    @store_time
    def find_corners(self):
        h, w = self.template_data.photo.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        corners = np.int32(cv2.perspectiveTransform(pts, self.transform))
        return corners

    @store_time
    def read_fields(self, field_data_df, model_df):
        crop_df = crop_sections(self.scan, field_data_df)
        self.field_df = tfw_wrapper.label_image_df(crop_df, model_df)
        return

    @store_time
    def evaluate_content(self, document_content_class):
        self.logic = document_content_class.from_fields(self.get_field_labels_dict())

    def get_field_labels(self):
        return self.field_df['label'] if self.field_df is not None else None

    def get_field_labels_json(self):
        return self.field_df['label'].to_json() if self.field_df is not None else None

    def get_field_labels_dict(self):
        return self.field_df['label'].to_dict() if self.field_df is not None else None

    def print_template_match_quality(self):
        print(str(len(self.template_data.keypoints)) + ' points in template, ' + str(len(self.image_data.keypoints)) + ' in photo, ' +
              str(len(self.matches)) + ' good matches')
        return

    def show_match_with_template(self):
        photo_with_match = self.image_data.photo.copy()

        if self.can_create_scan():
            h, w = self.template_data.photo.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, self.transform)
            cv2.polylines(photo_with_match, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - %d/%d" % (len(self.matches), MIN_MATCH_COUNT))

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=self.mask.ravel().tolist() if self.mask is not None else None,  # draw only inliers
                           flags=2)

        photo_with_match = cv2.drawMatches(self.template_data.photo, self.template_data.keypoints, photo_with_match, self.image_data.keypoints, self.matches, None, **draw_params)

        display(photo_with_match)
        return

    def show_scan(self):
        # show the original and scanned images
        display(resize(self.photo, 650), resize(self.scan, 650))

    def show_boxes(self, field_data_df):
        import random
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig = plt.figure(figsize=(16, 13))
        plt.imshow(self.scan)

        colours = [(1, 1, 1)] + [(random.random(), random.random(), random.random()) for i in range(255)]
        new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colours, N=256)
        for name, (l, r, u, d) in field_data_df['coords'].iteritems():
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
    
    def save_images_and_case_log(self, log_path, case_id):
        case_log = {}
        case_log['case_id'] = case_id
        case_log['log_path'] = log_path
        case_log['predictions'] = self.get_prediction_dict()
        case_log['method_times'] = self._method_times
        if getattr(self.logic, 'get_case_log', None) is not None:
            case_log['content'] = self.logic.get_case_log()
        
        case_log['image_paths'] = {}
        image_name_list = ['photo', 'photo_grey', 'scan']
        for image_name in image_name_list:
            image = getattr(self, image_name, None)
            if image is not None:
                image_output_path = os.path.join(log_path, '{}_{}.jpg'.format(case_id, image_name))
                case_log['image_paths'][image_name] = image_output_path
                cv2.imwrite(image_output_path, image)
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
        prediction_dict['document_error'] = getattr(self, 'error_reason')
        return prediction_dict
    
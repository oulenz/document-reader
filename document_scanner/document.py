from abc import ABC

import cv2
import numpy as np

import document_scanner.cv_wrapper as cv_wrapper
import document_scanner.tfw_wrapper as tfw_wrapper

MIN_MATCH_COUNT = 10

class Document(ABC):

    def __init__(self, img_path: str):
        self.img_path = img_path
        self.photo = cv2.imread(img_path, 1)
        self.photo_grey = cv2.imread(img_path, 0)
        self.error_reason = None
        #self.photo_grey = cv2.cvtColor(self.photo, cv2.COLOR_BGR2GRAY)

    @classmethod
    def as_template(cls, img_path:str, orb):
        document = cls(img_path)
        document.scan = document.photo
        document.resized = document.photo
        document.identify_keypoints(orb)
        return document

    def find_document_type(self, model_and_labels, pretrained_client = None):
        model, label_dict = model_and_labels
        self.document_type_name = tfw_wrapper.label_img(self.photo_grey, model, label_dict, pretrained_client)

    # canny edge method, not currently used
    def find_edges(self):
        self.edged = cv_wrapper.find_edges(self.photo_grey)
        return

    def identify_keypoints(self, orb):
        # Find the keypoints and descriptors.
        self.keypoints, self.kp_descriptors = cv_wrapper.get_keypoints_and_descriptors(self.resized, orb)

    def find_match(self, template, orb):
        self.template = template
        h, w = template.photo.shape[:2]
        height_to_use = int(h * 1.2)
        self.resized = cv_wrapper.resize(self.photo_grey, height_to_use)
        self.identify_keypoints(orb)
        self.matches = cv_wrapper.get_matching_points(template.kp_descriptors, self.kp_descriptors)
        self.good_matches = cv_wrapper.select_good_matches(self.matches)

    def can_create_scan(self):
        return len(self.matches) > MIN_MATCH_COUNT

    def create_scan(self):
        self.transform, self.mask = cv_wrapper.find_transformation_and_mask(self.template.keypoints, self.keypoints, self.good_matches)
        self.scan = cv_wrapper.reverse_transformation(self.resized, self.transform, self.template.photo.shape)
        return

    def read_document(self, field_data_df, model_df):
        field_df = cv_wrapper.crop_sections(self.scan, field_data_df)
        self.content_df = tfw_wrapper.label_image_df(field_df, model_df)
        return

    def get_content_labels(self):
        return self.content_df['label']

    def get_content_labels_json(self):
        return self.content_df['label'].to_json()

    def print_template_match_quality(self):
        print(str(len(self.template.keypoints)) + ' points in template, ' + str(len(self.keypoints)) + ' in photo, ' +
              str(len(self.matches)) + ' matches, ' + str(len(self.good_matches)) + ' good matches')
        return

    def show_match_with_template(self):
        photo_with_match = self.resized.copy()

        if self.can_create_scan():
            h, w = self.template.photo.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, self.transform)
            cv2.polylines(photo_with_match, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - %d/%d" % (len(self.good_matches), MIN_MATCH_COUNT))

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=self.mask.ravel().tolist() if self.mask is not None else None,  # draw only inliers
                           flags=2)

        photo_with_match = cv2.drawMatches(self.template.photo, self.template.keypoints, photo_with_match, self.keypoints, self.good_matches, None, **draw_params)

        cv_wrapper.display(photo_with_match)
        return

    def show_scan(self):
        # show the original and scanned images
        cv_wrapper.display(cv_wrapper.resize(self.photo, 650), cv_wrapper.resize(self.scan, 650))

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
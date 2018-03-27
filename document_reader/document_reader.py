import cv2
import inspect
import json
import os
import pandas as pd
import time

from abc import ABC
from typing import Dict, Optional, Set, Tuple, Union

from experiment_logger.loggable import ObjectDict

from document_reader.cv_wrapper import crop_sections, display, find_transform_and_mask, get_keypoints_and_descriptors, get_matching_points, get_orb, low_and_high_pass_filter, low_pass_filter, resize, reverse_transformation
from document_reader.document import Document, DocumentScan, Image_data
from document_reader.os_wrapper import list_subfolders
from document_reader.py_wrapper import startswith
from document_reader.tfs_wrapper import get_labeled_img_df, ImageClassifier

# TODO: re-add argument types Dict[str, ProdClient] and Union[str, ProdClient] once we've figured out how to import
# ProdClient here and InMemoryClient in tfs_wrapper without causing tensorflow to go bonkers.


class DocumentReader(ABC):

    document_class = Document

    def __init__(self):
        self.document_type_name_or_classifier = None
        self.field_classifier_dct = None
        self.field_data_df = None
        self.orb = None
        self.template_df = None
    
    @classmethod
    def from_dicts(cls, document_type_model_path: str, template_path_dct: Dict[str, str], model_dir: str, field_model_selection: Set[str], field_data_df: pd.DataFrame, document_type_name_or_client=None, field_client_dct: Dict = None):
        scanner = cls()
        scanner.orb = get_orb()
        scanner.template_df = scanner.parse_document_type_data(template_path_dct)
        scanner.document_type_name_or_classifier = cls.get_document_type_classifier(document_type_model_path, document_type_name_or_client)
        scanner.field_data_df = cls.parse_field_data(field_data_df, field_model_selection)
        scanner.field_classifier_dct = cls.get_field_classifier_dct(model_dir, field_model_selection, field_client_dct)
        return scanner

    @staticmethod
    def parse_field_data(field_data_df: pd.DataFrame, field_types: Set[str]):
        field_data_df = field_data_df[field_data_df['model_name'].isin(field_types)]
        field_data_df['lrud'] = field_data_df['lrud'].apply(lambda x: tuple([int(y) for y in x.split(':')]))  # (l, r, u, d)
        field_data_df = field_data_df.set_index(['document_type_name', 'field_name'])
        return field_data_df

    @staticmethod
    def get_document_type_classifier(document_type_model_path: str, document_type_name_or_client):
        if type(document_type_name_or_client) == str:
            return document_type_name_or_client
        with open(document_type_model_path + '.json', 'r') as f:
            objdct = ObjectDict(json.load(f))
        if document_type_name_or_client:
            return objdct.to_object(model_client=document_type_name_or_client)
        return objdct.to_object(model_path=document_type_model_path)

    @staticmethod
    def get_field_classifier_dct(model_directory: str, field_types: Set[str], field_client_dct: Dict):
        classifier_dct = {}

        for name in field_types:
            with open(os.path.join(model_directory, name + '.json'), 'r') as f:
                objdct = ObjectDict(json.load(f))
            if field_client_dct:
                classifier = objdct.to_object(model_client=field_client_dct[name])
            else:
                dirname = max([dirname for dirname in list_subfolders(model_directory) if startswith(dirname.split('_'), name.split('_'))])
                classifier = objdct.to_object(model_path=os.path.join(model_directory, dirname))
            classifier_dct[name] = classifier

        return classifier_dct

    def parse_document_type_data(self, template_path_dct: Dict[str, str]):
        document_type_df = pd.DataFrame(list(template_path_dct.items()), columns=['document_type_name', 'image_path'])
        def get_image_data(img_path):
            img = cv2.imread(img_path, 0)
            return Image_data.of_photo(img, self.orb)

        document_type_df['template'] = document_type_df['image_path'].apply(get_image_data)

        return document_type_df.set_index('document_type_name')

    def is_good_scan(self, scan: DocumentScan, document_type_name: str):
        return scan.result is not None

    def develop_document(self, photo) -> document_class:
        start_time = time.time()
        photo_grey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        document_type_name = self.predict_document_type(photo_grey, self.document_type_name_or_classifier)
        scan, retries = self.scan_document(photo_grey, document_type_name)
        field_df = self.read_fields(scan.result, self.field_data_df.xs(document_type_name))
        document = self.document_class.from_parts(photo=photo, document_type_name=document_type_name, scan=scan, field_df=field_df, scan_retries=retries)
        document._method_times.append((inspect.currentframe().f_code.co_name, time.time() - start_time))
        return document

    def scan_document(self, photo, document_type_name) -> Tuple[Optional[DocumentScan], Optional[int]]:
        template_data = self.template_df.loc[document_type_name, 'template']
        for i, img in enumerate(self.get_match_candidates(photo)):
            scan = DocumentScan.from_photo(img, template_data, self.orb)
            if self.is_good_scan(scan, document_type_name):
                return scan, i
        return None, None

    @staticmethod
    def predict_document_type(photo, document_type_name_or_classifier):
        if type(document_type_name_or_classifier) == str:
            return document_type_name_or_classifier
        return document_type_name_or_classifier.img_to_prediction(photo)

    def get_match_candidates(self, img):
        # ordered according to descending incremental usefulness in terms of creating good scans
        return img, low_and_high_pass_filter(img), low_pass_filter(img)

    def read_fields(self, scan_result, field_data_df):
        crop_df = crop_sections(scan_result, field_data_df)
        field_df = get_labeled_img_df(crop_df, self.field_classifier_dct)
        return field_df

import cv2
import inspect
import json
import os
import pandas as pd
import time

from abc import ABC
from typing import Dict, Set, Union

from experiment_logger.loggable import ObjectDict

from document_reader.cv_wrapper import get_orb
from document_reader.document import Document, Image_data
from document_reader.os_wrapper import list_subfolders
from document_reader.py_wrapper import startswith
from document_reader.tfs_wrapper import ImageClassifier

# TODO: re-add argument types Dict[str, ProdClient] and Union[str, ProdClient] once we've figured out how to import
# ProdClient here and InMemoryClient in tfs_wrapper without causing tensorflow to go bonkers.


class DocumentReader(ABC):

    def __init__(self):
        self.document_type_name_or_classifier = None
        self.field_classifier_dct = None
        self.field_data_df = None
        self.orb = None
        self.template_df = None
    
    @classmethod
    def from_dicts(cls, document_type_model_path: str, template_path_dct: Dict[str, str], model_dir: str, field_types: Set[str], field_data_df: pd.DataFrame, document_type_name_or_client=None, field_client_dct: Dict = None):
        scanner = cls()
        scanner.orb = get_orb()
        scanner.template_df = scanner.parse_document_type_data(template_path_dct)
        scanner.document_type_name_or_classifier = cls.get_document_type_classifier(document_type_model_path, document_type_name_or_client)
        scanner.field_data_df = cls.parse_field_data(field_data_df)
        scanner.field_classifier_dct = cls.get_field_classifier_dct(model_dir, field_types, field_client_dct)
        return scanner

    @staticmethod
    def parse_field_data(field_data_df: pd.DataFrame):
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

    def develop_document(self, img_path: str):
        start_time = time.time()
        document = Document.from_path(img_path)
        document.predict_document_type(self.document_type_name_or_classifier)
        if document.document_type_name not in self.template_df.index:
            document.error_reason = 'document_type'
            return document
        template_data = self.template_df.loc[document.document_type_name, 'template']
        document.template_data = template_data
        for i, img in enumerate(document.get_match_candidates(template_data)):
            document.find_match(img, template_data, self.orb)
            if not document.can_create_scan():
                continue
            document.find_transform_and_mask()
            document.create_scan()
            if document.scan is None:
                continue
            document.scan_retries = i
            break
        if document.scan is None:
            document.error_reason = 'image_quality'
            return document
        document.read_fields(self.field_data_df.xs(document.document_type_name), self.field_classifier_dct)
        document._method_times.append((inspect.currentframe().f_code.co_name, time.time() - start_time))
        return document




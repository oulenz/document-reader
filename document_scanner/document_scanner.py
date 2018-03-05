import cv2
import inspect
import json
import os
import pandas as pd
import time

from abc import ABC
from typing import Dict, Union

from experiment_logger.loggable import ObjectDict

from document_scanner.cv_wrapper import get_orb, pad_coords
from document_scanner.document import Document, Image_data
from document_scanner.py_wrapper import get_class_from_module_path
from document_scanner.tfs_wrapper import ImageClassifier

# TODO: re-add argument types Dict[str, ProdClient] and Union[str, ProdClient] once we've figured out how to import
# ProdClient here and InMemoryClient in tfs_wrapper without causing tensorflow to go bonkers.

PADDING = 8

class Document_scanner(ABC):

    def __init__(self):
        self.business_logic_class = None
        self.document_type_predicter = None
        self.field_data_df = None
        self.field_classifier_dct = None
        self.orb = None
        self.paths = None
        self.template_df = None
    
    @classmethod
    def for_creating_scans(cls, path_dict_path: str, document_type_client: Union[str, ImageClassifier]):
        scanner = cls()
        paths = cls.parse_path_dict(path_dict_path)
        scanner.orb = get_orb()
        scanner.document_type_predicter = document_type_client
        scanner.template_df = scanner.parse_document_type_data(paths['document_type_data'], paths['data_dir'])
        scanner.business_logic_class = get_class_from_module_path(paths['business_logic_class'])
        return scanner
    
    @classmethod
    def for_extracting_content(cls, path_dict_path: str, field_client_dct: Dict = None):
        scanner = cls()
        paths = cls.parse_path_dict(path_dict_path)
        scanner.field_data_df = cls.parse_field_data(paths['field_data'])
        scanner.field_classifier_dct = cls.get_field_classifier_dct(paths['model_data'], paths['model_dir'], field_client_dct)
        scanner.business_logic_class = get_class_from_module_path(paths['business_logic_class'])
        return scanner
    
    @classmethod
    def for_anything(cls, path_dict_path: str, document_type_client = None, field_client_dct: Dict = None):
        scanner = cls()
        paths = cls.parse_path_dict(path_dict_path)
        scanner.paths = paths
        scanner.orb = get_orb()
        scanner.document_type_predicter = cls.get_document_type_predicter(paths['model_data'], paths['model_dir'], document_type_client)
        scanner.field_data_df = cls.parse_field_data(paths['field_data'])
        scanner.field_classifier_dct = cls.get_field_classifier_dct(paths['model_data'], paths['model_dir'], field_client_dct)
        scanner.template_df = scanner.parse_document_type_data(paths['document_type_data'], paths['data_dir'])
        scanner.business_logic_class = get_class_from_module_path(paths['business_logic_class'])
        return scanner

    @staticmethod
    def parse_path_dict(path: str):
        paths = {}
        dir_path = os.path.split(path)[0]
        paths['data_dir'] = dir_path

        with open(path, 'r') as config_file:
            for line in config_file.read().splitlines():
                if line.startswith('#'):
                    continue # skip comments
                line = line.split('#', 1)[0] # strip comments
                k, v = line.split('=', 1)  # only consider first occurence of =
                paths[k] = os.path.join(dir_path, v)

        return paths

    @staticmethod
    def parse_field_data(field_data_path: str):
        field_data_df = pd.read_csv(field_data_path, delimiter='|', comment='#')
        field_data_df['coords'] = field_data_df['coords'].apply(lambda x: pad_coords(tuple([int(y) for y in x.split(':')]), PADDING))  # (l, r, u, d)
        field_data_df = field_data_df.set_index(['document_type_name', 'field_name'])
        return field_data_df

    @staticmethod
    def get_model_filename_dct(model_data_path):
        model_df = pd.read_csv(model_data_path, delimiter='|', comment='#').set_index('model_name')
        return model_df['filename'].to_dict()

    @classmethod
    def get_document_type_predicter(cls, model_data_path, model_dir_path, document_type_client):
        if document_type_client and type(document_type_client) == str:
            return document_type_client
        print(type(document_type_client))
        model_filename_dct = cls.get_model_filename_dct(model_data_path)
        filename = model_filename_dct['document_type']
        with open(os.path.join(model_dir_path, filename + '.json'), 'r') as f:
            objdct = ObjectDict(json.load(f))
        if document_type_client:
            return objdct.to_object(model_client=document_type_client)
        return objdct.to_object(model_path=os.path.join(model_dir_path, filename + '.pb'))

    @classmethod
    def get_field_classifier_dct(cls, model_data_path, model_dir_path, field_client_dct):
        model_df = pd.read_csv(model_data_path, delimiter='|', comment='#').set_index('model_name')
        model_filename_dct = model_df['filename'].to_dict()
        classifier_dct = {}

        for name, filename in model_filename_dct.items():
            with open(os.path.join(model_dir_path, filename + '.json'), 'r') as f:
                objdct = ObjectDict(json.load(f))
            if field_client_dct:
                classifier = objdct.to_object(model_client=field_client_dct[name])
            else:
                classifier = objdct.to_object(model_path=os.path.join(model_dir_path, filename))
            classifier_dct[name] = classifier

        return classifier_dct

    def parse_document_type_data(self, document_type_data_path: str, data_dir_path: str):
        document_type_df = pd.read_csv(document_type_data_path, delimiter='|', comment='#')
        def get_image_data(img_path):
            img = cv2.imread(img_path, 0)
            return Image_data.of_photo(img, self.orb)
        
        document_type_df['image_path'] = document_type_df['image_path'].apply(lambda x: os.path.join(data_dir_path, x))
        document_type_df['template'] = document_type_df['image_path'].apply(get_image_data)

        return document_type_df.set_index('document_type_name')

    def develop_document(self, img_path: str):
        start_time = time.time()
        document = Document.from_path(img_path, self.business_logic_class)
        document.predict_document_type(self.document_type_predicter)
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
            document.read_fields(self.field_data_df.xs(document.document_type_name), self.field_classifier_dct)
            document.evaluate_content(self.business_logic_class)
            if hasattr(document.logic, 'is_good_scan') and not getattr(document.logic, 'is_good_scan')():
                continue
            document.scan_retries = i
            break
        if document.scan is None:
            document.error_reason = 'image_quality'
            return document
        document._method_times.append((inspect.currentframe().f_code.co_name, time.time() - start_time))
        return document




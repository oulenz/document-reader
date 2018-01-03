import cv2
import inspect
import json
import os
import pandas as pd
import tensorflow as tf
import time

from abc import ABC
from document_scanner.cv_wrapper import get_orb, pad_coords
from document_scanner.document import Document, Image_data
from document_scanner.py_wrapper import get_class_from_module_path
from tfwrapper.models import TransferLearningModel
from tfwrapper.models.frozen import FrozenInceptionV4
from tfwrapper.models.nets import NeuralNet, ShallowCNN


PADDING = 8

class Document_scanner(ABC):

    def __init__(self):
        self.business_logic_class = None
        self.document_type_model_and_labels = None
        self.field_data_df = None
        self.inceptionv4_client = None
        self.mock_document_type_name = None
        self.model_df = None
        self.orb = None
        self.path_dict = None
        self.template_df = None
    
    @classmethod
    def for_document_identification(cls, path_dict_path: str, mock_document_type_name):
        scanner = cls()
        scanner.path_dict = cls.parse_path_dict(path_dict_path)
        scanner.orb = get_orb()
        scanner.mock_document_type_name = mock_document_type_name
        scanner.template_df = scanner.parse_document_type_data(scanner.path_dict['document_type_data_path'], scanner.path_dict['data_dir_path'])
        scanner.business_logic_class = get_class_from_module_path(scanner.path_dict['business_logic_class_path'])
        return scanner
    
    @classmethod
    def for_document_content(cls, path_dict_path: str):
        scanner = cls()
        scanner.path_dict = cls.parse_path_dict(path_dict_path)
        scanner.field_data_df = cls.parse_field_data(scanner.path_dict['field_data_path'])
        scanner.model_df = cls.parse_model_data(scanner.path_dict['model_data_path'], scanner.path_dict['data_dir_path'])
        scanner.business_logic_class = get_class_from_module_path(scanner.path_dict['business_logic_class_path'])
        return scanner
    
    @classmethod
    def complete(cls, path_dict_path: str, inceptionv4_client=None, mock_document_type_name=None):
        scanner = cls()
        scanner.path_dict = cls.parse_path_dict(path_dict_path)
        scanner.orb = get_orb()
        scanner.inceptionv4_client = inceptionv4_client
        scanner.mock_document_type_name = mock_document_type_name
        scanner.document_type_model_and_labels = None if mock_document_type_name else cls.parse_document_type_model(scanner.path_dict['document_type_model_path'], inceptionv4_client is not None)
        scanner.field_data_df = cls.parse_field_data(scanner.path_dict['field_data_path'])
        scanner.model_df = cls.parse_model_data(scanner.path_dict['model_data_path'], scanner.path_dict['data_dir_path'])
        scanner.template_df = scanner.parse_document_type_data(scanner.path_dict['document_type_data_path'], scanner.path_dict['data_dir_path'])
        scanner.business_logic_class = get_class_from_module_path(scanner.path_dict['business_logic_class_path'])
        return scanner

    @staticmethod
    def parse_path_dict(path: str):
        path_dict = {}
        dir_path = os.path.split(path)[0]
        path_dict['data_dir_path'] = dir_path
        with open(path, 'r') as config_file:
            for line in config_file.read().splitlines():
                if line.startswith('#'):
                    continue # skip comments
                line = line.split('#', 1)[0] # strip comments
                k, v = line.split('=', 1)  # only consider first occurence of =
                path_dict[k] = os.path.join(dir_path, v)

        return path_dict

    @staticmethod
    def parse_document_type_model(document_type_model_path: str, with_pretrained_client: bool = False):
        document_type_model_prediction_path = document_type_model_path + '_prediction.tw'
        with open(document_type_model_prediction_path) as f:
            model_config = json.load(f)
            label_dict = model_config['labels']
        if with_pretrained_client:
            model = NeuralNet.from_tw(document_type_model_prediction_path, sess=None)
        else:
            model = TransferLearningModel.from_tw(document_type_model_path)

        return (model, label_dict)

    @staticmethod
    def parse_field_data(field_data_path):
        field_data_df = pd.read_csv(field_data_path, delimiter='|', comment='#')
        field_data_df['coords'] = field_data_df['coords'].apply(lambda x: pad_coords(tuple([int(y) for y in x.split(':')]), PADDING))  # (l, r, u, d)
        field_data_df = field_data_df.set_index(['document_type_name', 'field_name'])
        return field_data_df

    @staticmethod
    def parse_model_data(model_data_path, data_dir_path):
        def load_model_and_labels(model_path):
            tf.reset_default_graph()

            with open(model_path + '.tw') as tw_file:
                model_config = json.load(tw_file)

            model_name = model_config['name']
            [num_labels] = model_config['y_shape']
            h, w, c = model_config['X_shape']
            with tf.Session() as sess:
                model = ShallowCNN([h, w, c], num_labels, sess=sess, name=model_name)
                model.load(model_path, sess=sess)

            label_dict = model_config['labels']

            return model, label_dict

        model_df = pd.read_csv(model_data_path, delimiter='|', comment='#').set_index(['document_type_name', 'model_name'])
        model_df['model_path'] = model_df['model_path'].apply(lambda x: os.path.join(data_dir_path, x))
        model_df['model'], model_df['label_dict'] = zip(*model_df['model_path'].map(load_model_and_labels))

        return model_df

    def parse_document_type_data(self, document_type_data_path, data_dir_path):
        document_type_df = pd.read_csv(document_type_data_path, delimiter='|', comment='#')
        def get_image_data(img_path):
            img = cv2.imread(img_path, 0)
            return Image_data.of_photo(img, self.orb)
        
        document_type_df['image_path'] = document_type_df['image_path'].apply(lambda x: os.path.join(data_dir_path, x))
        document_type_df['template'] = document_type_df['image_path'].apply(get_image_data)

        return document_type_df.set_index('document_type_name')

    def develop_document(self, img_path: str, debug: bool = False):
        start_time = time.time()
        document = Document.from_path(img_path, self.business_logic_class)
        document.predict_document_type(self.document_type_model_and_labels, self.inceptionv4_client, self.mock_document_type_name)
        if document.document_type_name not in self.template_df.index:
            document.error_reason = 'document_type'
            return document
        template_data = self.template_df.loc[document.document_type_name, 'template']
        document.template_data = template_data
        for i, img in enumerate(document.get_match_candidates(template_data)):
            print('test')
            document.find_match(img, template_data, self.orb)
            if not document.can_create_scan():
                continue
            document.find_transform_and_mask()
            document.create_scan()
            if document.scan is None:
                continue
            document.read_fields(self.field_data_df.xs(document.document_type_name), self.model_df.xs(document.document_type_name))
            document.evaluate_content(self.business_logic_class)
            if hasattr(document.logic, 'is_good_scan') and not getattr(document.logic, 'is_good_scan')():
                print(getattr(document.logic, 'is_good_scan')())
                continue
            print(getattr(document.logic, 'is_good_scan')())
            document.scan_retries = i
            break
        if document.scan is None:
            document.error_reason = 'image_quality'
            return document
        document._method_times.append((inspect.currentframe().f_code.co_name, time.time() - start_time))
        if debug:
            document.show_debug_information(self.field_data_df)
        return document



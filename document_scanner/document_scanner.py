import json
import os
from abc import ABC

import pandas as pd
import tensorflow as tf
#from tfwrapper.models.nets import CNN
from tfwrapper.models.nets import ShallowCNN

import document_scanner.cv_wrapper as cv_wrapper
from document_scanner.document import Document

PADDING = 8

class Document_scanner(ABC):

    def __init__(self, config_path: str):
        self.config = self.parse_config(config_path)
        self.orb = cv_wrapper.get_orb()
        self.template = Document.as_template(self.config['template_path'], self.orb)
        self.field_data_df = self.parse_field_data(self.config)
        self.model_dict = self.parse_model_data(self.config)

    @staticmethod
    def parse_config(path: str):
        config_dict = {}
        dir_path = os.path.split(path)[0]
        config_dict['data_dir_path'] = dir_path
        with open(path, 'r') as config_file:
            for line in config_file.read().splitlines():
                if line.startswith('#'):
                    continue # skip comments
                line = line.split('#', 1)[0] # strip comments
                k, v = line.split('=', 1)  # only consider first occurence of =
                config_dict[k] = os.path.join(dir_path, v)

        return config_dict

    @staticmethod
    def parse_field_data(config):
        field_data_df = pd.read_csv(config['field_data_path'], delimiter='|', comment='#')
        field_data_df['coords'] = field_data_df['coords'].apply(lambda x: cv_wrapper.pad_coords(tuple([int(y) for y in x.split(':')]), PADDING))  # (l, r, u, d)
        field_data_df = field_data_df.set_index('field_name')
        return field_data_df

    @staticmethod
    def parse_model_data(config):
        model_dict = (pd.read_csv(config['model_data_path'], delimiter='|', comment='#')
                      .set_index('model_name')['model_path']
                      .to_dict()
                      )

        for name, path in model_dict.items():
            path =  os.path.join(config['data_dir_path'], path)
            with open(path + '.tw') as f:
                model_config = json.load(f)
                model_name = model_config['name']
                [num_labels] = model_config['y_shape']
                h, w, c = model_config['X_shape']

                tf.reset_default_graph()

                with tf.Session() as sess:
                    model = ShallowCNN([h, w, c], num_labels, sess=sess, name=model_name)
                    model.load(path, sess=sess)
                    label_dict = model_config['labels']
                    model_dict[name] = (model, label_dict)

        return model_dict

    def develop_document(self, img_path: str, debug: bool = False):
        document = Document(img_path)
        document.find_match(self.template, self.orb)
        if debug:
            document.print_template_match_quality()
        if not document.can_create_scan():
            if debug:
                document.show_match_with_template()
            return document
        document.create_scan()
        if debug:
            document.show_match_with_template()
            document.show_scan()
            document.show_boxes(self.field_data_df)
        document.read_document(self.field_data_df, self.model_dict)
        if debug:
            print(document.get_content_labels_json())
        return document



# import the necessary packages
import csv

import cv2
import cv_wrapper
import json
import numpy as np
import os
import pandas as pd

from abc import ABC
from document import Document
from tfwrapper.models.nets import ShallowCNN
from typing import Callable
from os_wrapper import get_parent_dir_path

PADDING = 8
BASE_DIR_PATH =  get_parent_dir_path(os.path.realpath(__file__))

class Document_scanner(ABC):

    def __init__(self, config_path: str):
        self.config = self.parse_config(config_path)
        self.orb = cv_wrapper.get_orb()
        self.template = Document.as_template(self.config['template_path'], self.orb)
        self.field_data_df = self.parse_field_data(self.config, self.template.photo.shape)
        self.model_dict = self.parse_model_data(self.config)


    def parse_config(self, path: str):
        config = {}
        with open(path, 'r') as config_file:
            for line in config_file.read().splitlines():
                if line.startswith('#'):
                    continue # skip comments
                line = line.split('#', 1)[0] # strip comments
                k, v = line.split('=', 1)  # only consider first occurence of =
                config[k] = os.path.join(BASE_DIR_PATH, v)

        return config


    def parse_field_data(self, config, template_shape):

        field_data_df = pd.read_csv(config['field_data_path'], delimiter='|', comment='#')
        field_data_df['coords'] = field_data_df['coords'].apply(lambda x: cv_wrapper.pad_coords(tuple([int(y) for y in x.split(':')]), PADDING, template_shape))  # (l, r, u, d)

        field_data_df = field_data_df.set_index('field_name')


        return field_data_df


    def parse_model_data(self, config):
        model_dict = (pd.read_csv(config['model_data_path'], delimiter='|', comment='#')
                      .set_index('model_name')['model_path']
                      .to_dict()
                      )

        for name, path in model_dict.items():
            path =  os.path.join(BASE_DIR_PATH, path)
            with open(path + '.tw') as f:
                model_config = json.load(f)
                model_name = model_config['name']
                num_labels = model_config['y_size']
                h, w, c = model_config['X_shape']

                model = ShallowCNN([h, w, c], num_labels, name=model_name)
                model.load(path)
                label_dict = model_config['labels']
                model_dict[name] = (model, label_dict)

        return model_dict

    def process_document(self, img_path: str, debug: bool = False):
        document = Document(img_path)
        document.create_scan(self.template, self.orb)
        if False:#debug:
            document.print_template_match_quality()
            document.show_match_with_template()
            document.show_scan()
            document.show_boxes(self.field_data_df)
        document.read_document(self.field_data_df, self.model_dict)
        if debug:
            print(document.content_df['label'])

        return



import json
import logging.config
import os
import pandas as pd
import tensorflow as tf

from abc import ABC
from document_scanner.cv_wrapper import get_orb, pad_coords
from document_scanner.document import Document
from tfwrapper.models import TransferLearningModel
from tfwrapper.models.frozen import FrozenInceptionV4
from tfwrapper.models.nets import NeuralNet, ShallowCNN


PADDING = 8

class Document_scanner(ABC):

    def __init__(self, path_dict_path: str, inceptionv4_client = None, log_level = 'INFO'):
        logging.config.dictConfig(self.get_logging_config_dict(log_level))
        self.logger = logging.getLogger(__name__)
        self.path_dict = self.parse_path_dict(path_dict_path)
        self.orb = get_orb()
        self.inceptionv4_client = inceptionv4_client
        self.document_type_model_and_labels = self.parse_document_type_model(self.path_dict['document_type_model_path'], inceptionv4_client is not None)
        self.field_data_df = self.parse_field_data(self.path_dict['field_data_path'])
        self.model_df = self.parse_model_data(self.path_dict['model_data_path'], self.path_dict['data_dir_path'])
        self.template_dict = self.parse_document_type_data(self.path_dict['document_type_data_path'], self.path_dict['data_dir_path'])

    @staticmethod
    def get_logging_config_dict(log_level):
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s %(levelname)s %(name)s %(message)s'
                },
            },
            'handlers': {
                'default': {
                    'level': log_level,
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard'
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'standard',
                    'filename': 'apilog.log',
                    'maxBytes': 5000 * 1024,
                    'backupCount': 3
                }
            },
            'loggers': {
                '': {
                    'handlers': ['default', 'file'],
                    'level': log_level,
                    'propagate': True
                }
            }
        }

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
        document_type_df['template'] = document_type_df['template_path'].apply(lambda x: Document.as_template(os.path.join(data_dir_path, x), self.orb))

        return document_type_df.set_index('document_type_name')['template'].to_dict()

    def develop_document(self, img_path: str, debug: bool = False):
        self.logger.info('Start developing document %s', img_path)
        document = Document(img_path)
        document.find_document_type(self.document_type_model_and_labels, self.inceptionv4_client)
        if document.document_type_name not in self.template_dict.keys():
            document.error_reason = 'document_type'
            self.logger.info('Predicted document type as %s, which cannot be handled; aborting', document.document_type_name)
            return document
        self.logger.debug('Predicted document type as %s', document.document_type_name)
        document.find_match(self.template_dict[document.document_type_name], self.orb)
        if debug:
            document.print_template_match_quality()
        if not document.can_create_scan():
            if debug:
                document.show_match_with_template()
            document.error_reason = 'image_quality'
            self.logger.info('Identified insufficient points for template matching; aborting')
            return document
        self.logger.debug('Identified points for template matching')
        document.create_scan()
        self.logger.debug('Created scan from original photo')
        if debug:
            document.show_match_with_template()
            document.show_scan()
            document.show_boxes(self.field_data_df)
        document.read_document(self.field_data_df.xs(document.document_type_name), self.model_df.xs(document.document_type_name))
        if debug:
            print(document.get_content_labels_json())
        self.logger.info('Successfully extracted content from document %s', img_path)
        return document



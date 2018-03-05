import numpy as np
import pandas as pd
from abc import ABC
from typing import Dict, List

from experiment_logger.loggable import Loggable
from predict_client.inmemory_client import InMemoryClient

from document_scanner.cv_wrapper import reshape


class ImageClassifier(Loggable):
    def __init__(self, input_shape, dtype: str, model_client, labels: List[str], in_tensor_name: str = 'inputs', out_tensor_name: str = 'outputs'):
        self.input_shape = input_shape
        self.dtype = dtype
        self.model_client = model_client
        self.labels = labels
        self.in_tensor_name = in_tensor_name
        self.out_tensor_name = out_tensor_name

    @classmethod
    def from_object_dict(cls, dct: Dict, **kwargs):

        for attribute_name in ['input_shape', 'dtype', 'labels', 'in_tensor_name', 'out_tensor_name']:
            if attribute_name not in kwargs:
                kwargs[attribute_name] = dct.get(attribute_name)

        if 'model_client' not in kwargs:
            if 'model_path' in kwargs:
                kwargs['model_client'] = InMemoryClient(kwargs['model_path'])
                del kwargs['model_path']
            else:
                raise ValueError('ImageClassificationPipeline.from_object_dict must be called with model_client or model_path')

        return super().from_object_dict(dct, **kwargs)

    def img_to_label(self, img):
        return self.imgs_to_labels([img])[0]

    def imgs_to_labels(self, imgs):
        x = self.imgs_to_x(imgs)
        y = self.x_to_y(x)
        return self.y_to_labels(y)

    def imgs_to_x(self, imgs):
        return np.array([reshape(img, self.input_shape) for img in imgs])

    def x_to_y(self, x):
        req = self._x_to_request(x)
        resp = self.model_client.predict(req)
        return self._response_to_y(resp)

    def _x_to_request(self, x):
        return [{'in_tensor_name': self.in_tensor_name, 'in_tensor_dtype': self.dtype, 'data': x}]

    def _response_to_y(self, resp):
        return resp[self.out_tensor_name]

    def y_to_labels(self, y):
        return [self.labels[np.argmax(y_i)] for y_i in y]

    def to_object_dict(self, **kwargs) -> Dict:

        for attribute_name in ['input_shape', 'dtype', 'labels', 'in_tensor_name', 'labels', 'out_tensor_name']:
            if attribute_name not in kwargs:
                kwargs[attribute_name] = getattr(self, attribute_name)

        return super().to_object_dict(**kwargs)


def get_labeled_img_df(img_df, model_dct):
    field_df = pd.DataFrame()

    for model_name, fields_of_model_df in img_df.groupby(['model_name']):
        model = model_dct[model_name]
        #input_shape, model, label_dict = model_df.loc[model_name, ['input_shape', 'model', 'label_dict']]
        #labels = imgs_to_labels(fields_of_model_df['crop'].values, input_shape, model, label_dict)
        labels = model.imgs_to_labels(fields_of_model_df['crop'].values)
        new_content = pd.DataFrame(fields_of_model_df['crop'].copy())
        new_content['label'] = labels
        field_df = field_df.append(new_content)

    return field_df

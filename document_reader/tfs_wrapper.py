import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from experiment_logger.loggable import Loggable

from document_reader.cv_wrapper import reshape


class ImageReader(Loggable):
    def __init__(self, *, in_reshape: Tuple[int, int, int] = (None, None, None), in_tensor_name: str = 'inputs', in_dtype: str, model_client, out_tensor_names: Tuple[str, ...], labels: List[str]):
        self.in_reshape = in_reshape
        self.in_dtype = in_dtype
        self.model_client = model_client
        self.labels = labels
        self.in_tensor_name = in_tensor_name
        self.out_tensor_names = out_tensor_names

    @classmethod
    def from_object_dict(cls, dct: Dict, **kwargs):

        for attribute_name in ['in_reshape', 'in_dtype', 'labels', 'in_tensor_name']:
            if attribute_name not in kwargs:
                kwargs[attribute_name] = dct.get(attribute_name)

        for attribute_name in ['in_reshape', 'out_tensor_names']:
            if attribute_name not in kwargs:
                kwargs[attribute_name] = tuple(dct.get(attribute_name))

        if 'model_client' not in kwargs:
            if 'model_path' in kwargs:
                from predict_client.inmemory_client import InMemoryClient
                kwargs['model_client'] = InMemoryClient(kwargs['model_path'])
                del kwargs['model_path']
            else:
                raise ValueError('from_object_dict must be called with model_client or model_path')

        return super().from_object_dict(dct, **kwargs)

    def to_object_dict(self, **kwargs) -> Dict:

        for attribute_name in ['in_reshape', 'in_dtype', 'labels', 'in_tensor_name', 'out_tensor_names']:
            if attribute_name not in kwargs:
                kwargs[attribute_name] = getattr(self, attribute_name)

        return super().to_object_dict(**kwargs)

    def img_to_prediction(self, img):
        return self.imgs_to_predictions([img])[0]

    def imgs_to_predictions(self, imgs):
        x = self.imgs_to_x(imgs)
        y = self.x_to_y(x)
        return self.y_to_predictions(y)

    def imgs_to_x(self, imgs):
        return np.array([reshape(img, self.in_reshape) for img in imgs])

    def x_to_y(self, x):
        req = self._x_to_request(x)
        # TODO: add escalating retries, set base value per class or object
        resp = self.model_client.predict(req, request_timeout=600)
        return self._response_to_y(resp)

    def _x_to_request(self, x):
        return [{'in_tensor_name': self.in_tensor_name, 'in_tensor_dtype': self.in_dtype, 'data': x}]

    def _response_to_y(self, resp):
        for tensor_name in self.out_tensor_names:
            if tensor_name not in resp:
                return tuple([[], [], []])

        return tuple([resp[tensor_name] for tensor_name in self.out_tensor_names])

    def y_to_predictions(self, y):
        return [self.y_to_prediction(*y_i) for y_i in zip(*y)]

    @abstractmethod
    def y_to_prediction(self, *y):
        pass


class ImageClassifier(ImageReader):

    def __init__(self, *, out_tensor_name: str = 'outputs', **kwargs):
        super().__init__(out_tensor_names=(out_tensor_name,), **kwargs)

    def y_to_prediction(self, y):
        return self.labels[np.argmax(y)]


def get_labeled_img_df(img_df, model_dct):
    field_df = pd.DataFrame()

    for model_name, fields_of_model_df in img_df.groupby(['model_name']):
        model = model_dct[model_name]
        labels = model.imgs_to_predictions(fields_of_model_df['crop'].values)
        new_content = pd.DataFrame(fields_of_model_df['crop'].copy())
        new_content['label'] = labels
        field_df = field_df.append(new_content)

    return field_df
import os
import json
import numpy as np
import tensorflow as tf

from document_scanner import Document_scanner

from utils import curr_path
from utils import remove_dir

def test_config_data(field_data_df):
    required_columns = {'name', 'coords', 'type', 'model_path', 'uses_inception', 'num_classes', 'crop'}

    for column in required_columns - set(field_data_df.columns):
        print('Error: required column ' + column + ' missing in data files')

def test_process_document():
    ds = Document_scanner()
    ds.process_document()

import os

from document_scanner.cv_wrapper import get_orb
from document_scanner.document import Document
from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import BASE_DIR_PATH

#from utils import curr_path
#from utils import remove_dir

CONFIG_FILE_PATH = os.path.join(BASE_DIR_PATH, 'data', 'config.txt')

def test_data_files():
    assert os.path.exists(CONFIG_FILE_PATH)

    config_dict = Document_scanner.parse_config(CONFIG_FILE_PATH)
    required_config_keys = {'template_path', 'field_data_path', 'model_data_path', 'models_path'}
    assert not required_config_keys - set(config_dict.keys())

    assert os.path.exists(config_dict['template_path'])
    assert os.path.exists(config_dict['field_data_path'])
    assert os.path.exists(config_dict['model_data_path'])
    assert os.path.exists(config_dict['models_path'])

    field_data_df = Document_scanner.parse_field_data(config_dict)
    required_field_data_df_columns = {'coords', 'model_name'}
    assert not required_field_data_df_columns - set(field_data_df.columns)

    model_dict = Document_scanner.parse_model_data(config_dict)
    assert model_dict is not None

    orb = get_orb()
    template = Document.as_template(config_dict['template_path'], orb)
    assert template.photo is not None
    assert template.keypoints is not None
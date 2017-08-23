import os

from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import BASE_DIR_PATH

PATH_DICT_PATH = os.path.join(BASE_DIR_PATH, 'data', 'paths.txt')

def test_data_paths():
    assert os.path.exists(PATH_DICT_PATH)

    config_dict = Document_scanner.parse_path_dict(PATH_DICT_PATH)
    required_config_keys = {'data_dir_path', 'document_type_model_path', 'field_data_path', 'model_data_path', 'document_type_data_path'}
    assert not required_config_keys - set(config_dict.keys())

    assert os.path.exists(config_dict['data_dir_path'])
    assert os.path.exists(config_dict['field_data_path'])
    assert os.path.exists(config_dict['model_data_path'])
    assert os.path.exists(config_dict['document_type_data_path'])


def test_initialisation():
    scanner = Document_scanner(PATH_DICT_PATH)

    required_field_data_df_columns = {'coords', 'model_name'}
    assert not required_field_data_df_columns - set(scanner.field_data_df.columns)

    required_model_df_columns = {'model', 'label_dict'}
    assert not required_model_df_columns - set(scanner.model_df.columns)

    for template in scanner.template_dict.values():
        assert template.photo is not None
        assert template.keypoints is not None

def test_develop_document():
    scanner = Document_scanner(PATH_DICT_PATH)
    for document_type_name, template in scanner.template_dict.items():
        document = scanner.develop_document(template.img_path)
        assert document.document_type_name == document_type_name
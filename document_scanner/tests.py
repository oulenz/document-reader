import os

from document_scanner.document import Document
from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH, remove_dir
from predict_client.mock_client import MockPredictClient


def test_data_paths():
    assert os.path.exists(DEFAULT_PATH_DICT_PATH)

    config_dict = Document_scanner.parse_path_dict(DEFAULT_PATH_DICT_PATH)
    required_config_keys = {'data_dir_path', 'document_type_model_path', 'field_data_path', 'model_data_path', 'document_type_data_path'}
    assert not required_config_keys - set(config_dict.keys())

    assert os.path.exists(config_dict['data_dir_path'])
    assert os.path.exists(config_dict['field_data_path'])
    assert os.path.exists(config_dict['model_data_path'])
    assert os.path.exists(config_dict['document_type_data_path'])


def test_initialisation():
    scanner = Document_scanner.complete(DEFAULT_PATH_DICT_PATH)

    required_field_data_df_columns = {'coords', 'model_name'}
    assert not required_field_data_df_columns - set(scanner.field_data_df.columns)

    required_model_df_columns = {'model', 'label_dict'}
    assert not required_model_df_columns - set(scanner.model_df.columns)

    for document_type_name in scanner.template_df.index:
        template = scanner.template_df.loc[document_type_name, 'template']
        assert template.photo is not None
        assert template.keypoints is not None


def test_develop_document():
    scanner = Document_scanner.complete(DEFAULT_PATH_DICT_PATH)
    for document_type_name in scanner.template_df.index:
        document = scanner.develop_document(scanner.template_df.loc[document_type_name, 'image_path'])
        assert document.document_type_name == document_type_name


def test_pretrained_client():
    incv4_client = MockPredictClient('localhost:9001', 'incv4', 1, num_scores=1536)
    scanner = Document_scanner.complete(DEFAULT_PATH_DICT_PATH, inceptionv4_client=incv4_client)
    for document_type_name in scanner.template_df.index:
        document = scanner.develop_document(scanner.template_df.loc[document_type_name, 'image_path'])


def test_mock_document_type_name():
    incv4_client = MockPredictClient('localhost:9001', 'incv4', 1, num_scores=1536)
    scanner = Document_scanner.complete(DEFAULT_PATH_DICT_PATH, inceptionv4_client=incv4_client, mock_document_type_name='standard')
    for document_type_name in scanner.template_df.index:
        document = scanner.develop_document(scanner.template_df.loc[document_type_name, 'image_path'])
        assert document.document_type_name == document_type_name

        
def test_save_images_and_case_log():
    scanner = Document_scanner.complete(DEFAULT_PATH_DICT_PATH)
    temp_dir = os.path.join(scanner.path_dict['data_dir_path'], 'temp')
    if not os.path.isdir(temp_dir):
        for document_type_name in scanner.template_df.index:
            document = scanner.develop_document(scanner.template_df.loc[document_type_name, 'image_path'])
            log_dir = os.path.join(temp_dir, document_type_name)
            os.makedirs(log_dir)
            document.save_images_and_case_log(log_dir, document_type_name)
        remove_dir(temp_dir)
    else:
        assert False


def test_save_images_and_case_log_with_empty_document():
    document = Document()
    temp_dir = os.path.join(os.path.split(DEFAULT_PATH_DICT_PATH)[0], 'temp')
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
        document.save_images_and_case_log(log_path=temp_dir, case_id='test')
        remove_dir(temp_dir)
    else:
        assert False

import os

from document_scanner.document import Document
from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH, remove_dir


def test_data_paths():
    assert os.path.exists(DEFAULT_PATH_DICT_PATH)

    paths = Document_scanner.parse_path_dict(DEFAULT_PATH_DICT_PATH)
    required_paths = {'data_dir', 'document_type_model', 'field_data', 'model_data', 'document_type_data'}
    assert not required_paths - set(paths.keys())

    assert os.path.exists(paths['data_dir'])
    assert os.path.exists(paths['field_data'])
    assert os.path.exists(paths['model_data'])
    assert os.path.exists(paths['document_type_data'])


def test_initialisation():
    scanner = Document_scanner.for_anything(DEFAULT_PATH_DICT_PATH, document_type_client='standard')

    required_field_data_df_columns = {'coords', 'model_name'}
    assert not required_field_data_df_columns - set(scanner.field_data_df.columns)

    # TODO: add check whether all required models are present

    for document_type_name in scanner.template_df.index:
        template = scanner.template_df.loc[document_type_name, 'template']
        assert template.photo is not None
        assert template.keypoints is not None


def test_develop_document():
    scanner = Document_scanner.for_anything(DEFAULT_PATH_DICT_PATH, document_type_client='standard')
    for document_type_name in scanner.template_df.index:
        document = scanner.develop_document(scanner.template_df.loc[document_type_name, 'image_path'])
        assert document.document_type_name == document_type_name


def test_document_type_client():
    # TODO: write test
    pass


def test_mock_document_type_name():
    # TODO: rewrite test
    scanner = Document_scanner.for_anything(DEFAULT_PATH_DICT_PATH, document_type_client='standard')
    for document_type_name in scanner.template_df.index:
        document = scanner.develop_document(scanner.template_df.loc[document_type_name, 'image_path'])
        assert document.document_type_name == document_type_name

        
def test_save_images_and_case_log():
    scanner = Document_scanner.for_anything(DEFAULT_PATH_DICT_PATH, document_type_client='standard')
    temp_dir = os.path.join(scanner.paths['data_dir'], 'temp')
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

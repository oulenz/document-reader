import json
import os

from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH


def generate(field_labels_path, path_dict_path):
    path_dict_path = path_dict_path or DEFAULT_PATH_DICT_PATH

    with open(field_labels_path) as f:
        field_label_dict = json.load(f)
    testset_field_label_dict = field_label_dict['testset']
    response_code_dict = {}
    scanner = Document_scanner.for_document_content(path_dict_path)

    for filename, document_field_label_dict in testset_field_label_dict.items():
        document_logic = scanner.business_logic_class.from_fields(document_field_label_dict)
        response_code_dict[filename] = document_logic.get_response_code()

    folder_path, _ = os.path.split(field_labels_path)
    with open(os.path.join(folder_path, 'response_codes.json'), 'w') as f:
        json.dump(response_code_dict, f, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate response codes from field labels')
    parser.add_argument('field_labels_path', help='The location of the json file with the field labels')
    parser.add_argument('--path_dict_path', help='The location of the path dict')
    args = parser.parse_args()
    generate(args.field_labels_path, args.path_dict_path)

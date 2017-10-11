import json
import os
from uuid import uuid4

from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH, list_images


def generate_predictions(testset_path, document_type_name, predictions_name, path_dict_path):
    path_dict_path = path_dict_path or DEFAULT_PATH_DICT_PATH

    scanner = Document_scanner.complete(path_dict_path, mock_document_type_name=document_type_name)

    predictions = dict()
    predictions['document_type_name'] = document_type_name
    predictions['testset'] = dict()

    for img_name in list_images(testset_path):
            print('Developing {}'.format(img_name)) 
            document = scanner.develop_document(os.path.join(testset_path, img_name))
            predictions['testset'][img_name] = document.get_field_labels_dict()

    file_name = (predictions_name or 'predictions-' + uuid4().hex) + '.json'
    print('Writing to: ' + file_name)

    with open(os.path.join(testset_path, file_name), 'w') as file:
        json.dump(predictions, file, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predict labels for a testset of documents')
    parser.add_argument('testset_path', help='The location of the testset')
    parser.add_argument('document_type_name', help='The document type')
    parser.add_argument('predictions_name', help='Name to be used for the predictions filename', nargs='?')
    parser.add_argument('--path_dict_path', help='The location of the path dict')
    args = parser.parse_args()
    generate_predictions(args.testset_path, args.document_type_name, args.predictions_name, args.path_dict_path)

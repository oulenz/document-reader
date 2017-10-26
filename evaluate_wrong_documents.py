import json
import os
from shutil import copyfile

from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH


def evaluate(predictions_path, path_dict_path):
    path_dict_path = path_dict_path or DEFAULT_PATH_DICT_PATH

    with open(predictions_path) as f:
        predictions = json.load(f)

    testset_predictions = predictions['testset']

    scanner = Document_scanner.for_document_content(path_dict_path)

    documents_with_results = []
    documents_without_results = []

    for filename, document_predictions in testset_predictions.items():
        document_predictions = testset_predictions[filename]

        content_predicted = scanner.business_logic_class.from_fields(document_predictions)

        document_has_result = False
        for key, predicted_value in content_predicted.results.items():
            if predicted_value:
                print('WARNING: {} has result {} {}'.format(filename, key, str(predicted_value)))
                document_has_result = True
        
        if document_has_result:
            documents_with_results.append(filename)
        else:
            documents_without_results.append(filename)
    
    print('Documents with results: {} / {} ({})'.format(len(documents_with_results), len(documents_without_results), len(documents_with_results)/len(documents_without_results)))
    
    predictions_folder, predictions_filename = os.path.split(predictions_path)
    predictions_name, _ = os.path.splitext(predictions_filename)
    eval_store_path = os.path.join(os.path.split(predictions_path)[0], 'evaluation_{}'.format(predictions_name))
    documents_path = predictions_folder
    documents_with_results_path = os.path.join(eval_store_path, 'documents_with_results')
    os.makedirs(documents_with_results_path)

    for filename in documents_with_results:
        copyfile(os.path.join(documents_path, filename), os.path.join(documents_with_results_path, filename))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Check that predicted labels of a testset doesn\'t give any results')
    parser.add_argument('predictions_path', help='The location of the json file with the predictions')
    parser.add_argument('--path_dict_path', help='The location of the path dict')
    args = parser.parse_args()
    evaluate(args.predictions_path, args.path_dict_path)

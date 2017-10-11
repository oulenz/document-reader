import json
import os
import numpy as np
from shutil import copyfile

from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH
from document_scanner.py_wrapper import aggregate_keys


def check_fields(ground_truth, predictions, fields, positive_labels):
    actual_positivity = {k: ground_truth[k] in positive_labels for k in fields}
    predicted_positivity = {k: predictions[k] in positive_labels for k in fields}

    true_negatives = {k for k in fields if not actual_positivity[k] and not predicted_positivity[k]}
    false_negatives = {k for k in fields if actual_positivity[k] and not predicted_positivity[k]}
    false_positives = {k for k in fields if not actual_positivity[k] and predicted_positivity[k]}
    true_positives = {k for k in fields if actual_positivity[k] and predicted_positivity[k]}

    incremental_confusion = [[len(true_negatives), len(false_positives)], [len(false_negatives), len(true_positives)]]
    return false_negatives, false_positives, incremental_confusion

def compare_result_dicts(predicted_result, ground_truth_result):
    result_missing = False
    result_wrong = False
    
    for key, ground_truth_value in ground_truth_result.items():
        if ground_truth_value and not predicted_result.get(key):
            result_missing = True
    
    for key, predicted_value in predicted_result.items():
        if predicted_value and predicted_value != ground_truth_result.get(key):
            result_wrong = True
    
    return result_missing, result_wrong


def evaluate(predictions_path, ground_truth_path, path_dict_path):
    path_dict_path = path_dict_path or DEFAULT_PATH_DICT_PATH
    
    with open(predictions_path) as f:
        predictions = json.load(f)
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    
    document_type_name = predictions['document_type_name']
    if document_type_name != ground_truth['document_type_name']:
        print('Warning: comparing labels of {} and {} document types'.format(document_type_name, ground_truth['document_type_name']))
    
    testset_predictions = predictions['testset']
    testset_ground_truth = ground_truth['testset']
    if len(testset_ground_truth) != len(testset_predictions):
        print('Warning: not checking labels of entire testset')

    scanner = Document_scanner.for_document_content(path_dict_path)
    model_fields = aggregate_keys(scanner.field_data_df['model_name'].xs(document_type_name).to_dict())
    model_positive_labels = {k: eval(v) for k, v in scanner.model_df.xs(document_type_name)['positive_labels'].items()}

    perfect_matches = []
    total_failure = []

    documents_with_wrong_results = []
    documents_with_missing_results = []
    documents_with_correct_results = []

    false_positive_fields = dict()
    false_negative_fields = dict()
    perfect_fields = dict()
    confusion_matrices = dict()
    
    for field_type in model_fields:
        false_positive_fields[field_type] = dict()
        false_negative_fields[field_type] = dict()
        perfect_fields[field_type] = set()
        confusion_matrices[field_type] = np.array([[0, 0], [0, 0]])

    for filename, document_predictions in testset_predictions.items():
        document_ground_truth = testset_ground_truth[filename]
        document_predictions = testset_predictions[filename]
        
        all_field_types_right = True
        all_field_types_wrong = True
        
        for field_type, fields in model_fields.items():
            false_negatives, false_positives, incremental_confusion = check_fields(document_ground_truth,
                                                                                   document_predictions,
                                                                                   fields,
                                                                                   model_positive_labels[field_type]
                                                                                   )

            if len(false_positives) > 0:
                false_positive_fields[field_type][filename] = false_positives
                all_field_types_right = False
            
            if len(false_negatives) > 0:
                false_negative_fields[field_type][filename] = false_negatives
                all_field_types_right = False
    
            if len(false_positives) == 0 and len(false_negatives) == 0:
                perfect_fields[field_type].add(filename)
                all_field_types_wrong = False

            confusion_matrices[field_type] = np.add(confusion_matrices[field_type], incremental_confusion)

        if all_field_types_right:
            perfect_matches.append(filename)

        if all_field_types_wrong:
            total_failure.append(filename)

        content_ground_truth = scanner.business_logic_class.from_fields(document_ground_truth)
        content_predicted = scanner.business_logic_class.from_fields(document_predictions)

        result_missing, result_wrong = compare_result_dicts(content_predicted.results, content_ground_truth.results)

        if result_missing:
            documents_with_missing_results.append(filename)
            
        if result_wrong:
            documents_with_wrong_results.append(filename)
        
        if not result_missing and not result_wrong:
            documents_with_correct_results.append(filename)

    response_list = []
    response_list.append('---- Confusion matrices ----')
    for field_type, confusion_matrix in confusion_matrices.items():
        response_list.append(field_type)
        response_list.append(str(confusion_matrix))

    response_list.append('')

    response_list.append('---- Percentages ----')
    for field_type, perfect_list in perfect_fields.items():
        response_list.append('Forms with perfect fiels of type {}'.format(field_type))
        response_list.append(str(len(perfect_list) / len(testset_predictions)))
    response_list.append('Forms with all field types perfect')
    response_list.append(str(len(perfect_matches) / len(testset_predictions)))

    response_list.append('')

    response_list.append('---- Documents with wrong results ----')
    response_list.append(str(len(documents_with_wrong_results) / len(testset_predictions)) + ' ' + str(len(documents_with_wrong_results)))
    response_list.append(str(sorted(documents_with_wrong_results)))
    response_list.append('---- Documents with missing results ----')
    response_list.append(str(len(documents_with_missing_results) / len(testset_predictions)) + ' ' + str(len(documents_with_missing_results)))
    response_list.append(str(sorted(documents_with_missing_results)))

    response_list.append('')
    response_list.append('---- Documents with correct results ----')
    response_list.append(str(len(documents_with_correct_results) / len(testset_predictions)) + ' ' + str(len(documents_with_correct_results)))

    response = '\n'.join(response_list)
    print(response)

    predictions_folder, predictions_filename = os.path.split(predictions_path)
    response_path = os.path.join(predictions_folder, 'evaluation')
    with open(response_path, 'w') as f:
        f.write(response)
    
    predictions_name, _ = os.path.splitext(predictions_filename)
    eval_store_path = os.path.join(os.path.split(predictions_path)[0], 'evaluation_{}'.format(predictions_name))
    documents_path = predictions_folder

    perfect_store_path = os.path.join(eval_store_path, 'perfect_fields')
    false_negatives_path = os.path.join(eval_store_path, 'false_negatives')
    false_positives_path = os.path.join(eval_store_path, 'false_positives')

    documents_with_wrong_results_path = os.path.join(eval_store_path, 'documents_with_wrong_results_path')
    documents_with_missing_results_path = os.path.join(eval_store_path, 'documents_with_missing_results_path')
    documents_with_correct_results_path = os.path.join(eval_store_path, 'documents_with_correct_results_path')

    os.makedirs(eval_store_path)
    os.makedirs(documents_with_wrong_results_path)
    os.makedirs(documents_with_missing_results_path)
    os.makedirs(documents_with_correct_results_path)
    
    for field_type, perfect_list in perfect_fields.items():
        os.makedirs(os.path.join(perfect_store_path, field_type))
        for filename in perfect_list:
            copyfile(os.path.join(documents_path, filename), os.path.join(perfect_store_path, field_type, filename))
    
    for field_type, false_negative_dict in false_negative_fields.items():
        os.makedirs(os.path.join(false_negatives_path, field_type))
        for filename, false_negatives in false_negative_dict.items():
            name, ext = os.path.splitext(filename)
            copyfile(os.path.join(documents_path, filename),
                     os.path.join(false_negatives_path, field_type, '-'.join([name] + sorted(false_negatives)) + ext))
    
    for field_type, false_positive_dict in false_positive_fields.items():
        os.makedirs(os.path.join(false_positives_path, field_type))
        for filename, false_positives in false_positive_dict.items():
            name, ext = os.path.splitext(filename)
            copyfile(os.path.join(documents_path, filename),
                     os.path.join(false_positives_path, field_type, '-'.join([name] + sorted(false_positives)) + ext))

    for filename in documents_with_wrong_results:
        copyfile(os.path.join(documents_path, filename), os.path.join(documents_with_wrong_results_path, filename))

    for filename in documents_with_missing_results:
        copyfile(os.path.join(documents_path, filename), os.path.join(documents_with_missing_results_path, filename))

    for filename in documents_with_correct_results:
        copyfile(os.path.join(documents_path, filename), os.path.join(documents_with_correct_results_path, filename))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare predicted labels of a testset with ground truth')
    parser.add_argument('predictions_path', help='The location of the json file with the predictions')
    parser.add_argument('ground_truth_path', help='The location of the json file with the ground truth')
    parser.add_argument('--path_dict_path', help='The location of the path dict')
    args = parser.parse_args()
    evaluate(args.predictions_path, args.ground_truth_path, args.path_dict_path)

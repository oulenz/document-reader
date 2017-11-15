import argparse
import cv2
import os

from document_scanner.cv_wrapper import crop_sections
from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH, list_images, list_image_branches, list_subfolder_branches
from document_scanner.py_wrapper import rstrip

def process(src_path: str, mock_document_type_name: str, model_names, with_subfolders: bool, path_dict_path: str,) -> None:
    path_dict_path = path_dict_path or DEFAULT_PATH_DICT_PATH
        
    folder_path = os.path.split(src_path)[0]

    if not os.path.isdir(src_path):
        image_branches = [os.path.split(src_path)[1]]
    elif with_subfolders:
        image_branches = list_image_branches(src_path)
    else:
        image_branches = list_images(src_path)

    scanner = Document_scanner.for_document_content(path_dict_path)

    crops_path = os.path.join(folder_path, 'crops')
    for model_name in model_names or scanner.model_df.xs(mock_document_type_name).index:
        for subfolder_branch in list_subfolder_branches(folder_path):
            os.makedirs(os.path.join(crops_path, model_name, subfolder_branch), exist_ok=True)

    field_data_df = scanner.field_data_df.xs(mock_document_type_name)
    if model_names:
        field_data_df = field_data_df[field_data_df['model_name'].isin(model_names)]

    file_ext = '.jpg'  # always save as jpg to simplify tensorflow processing
    for image_branch in image_branches:
        print(image_branch)
        subfolder_branch, image_filename = os.path.split(image_branch)
        image_name = rstrip(os.path.splitext(image_filename)[0], '_scan')
        
        scan = cv2.imread(os.path.join(folder_path, image_branch), 0)
        field_df = crop_sections(scan, field_data_df)

        for field_name, row in field_df.iterrows():
            crop_filename = os.path.join(crops_path, row['model_name'], subfolder_branch, image_name + '_' + field_name + file_ext)
            cv2.imwrite(crop_filename, row['crop'])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop and store boxes from a folder of images')
    parser.add_argument('scan_path', help='The location of a scan or a folder of scans')
    parser.add_argument('mock_document_type_name', help='mock_document_type_name to use')
    parser.add_argument('--model_names', dest='model_names', nargs='*', help='Models to crop fields for')
    parser.add_argument('--with_subfolders', dest='with_subfolders', action='store_true', help='Also do images in subfolders')
    parser.add_argument('--path_dict_path', help='The location of the path file')
    args = parser.parse_args()
    process(args.scan_path, args.mock_document_type_name, args.model_names, args.with_subfolders, args.path_dict_path)
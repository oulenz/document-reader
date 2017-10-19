import cv2
import json
import os

from document_scanner.document import Document
from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH, list_images


def process(src_path: str, path_dict_path: str, document_type_name: str, create_scan: bool, create_dataset: bool, move_processed: bool) -> None:
    path_dict_path = path_dict_path or DEFAULT_PATH_DICT_PATH
        
    folder_path = src_path if os.path.isdir(src_path) else os.path.split(src_path)[0]
    img_filenames = list_images(src_path) if os.path.isdir(src_path) else [os.path.split(src_path)[1]]
    
    if create_scan:
        scan_folder_path = os.path.join(folder_path, 'scans')
        os.makedirs(scan_folder_path, exist_ok=True)
    if create_dataset:
        dataset_folder_path = os.path.join(folder_path, 'dataset')
        os.makedirs(dataset_folder_path, exist_ok=True)
        dataset_dict = {}
    
    scanner = Document_scanner.for_document_identification(path_dict_path, mock_document_type_name=document_type_name)
        
    for img_filename in img_filenames:
        print(img_filename)

        document = Document.from_path(os.path.join(folder_path, img_filename), scanner.business_logic_class)
        document.find_match(scanner.template_df.loc[document_type_name, 'template'], scanner.orb)
        document.find_transform_and_mask()
        
        img_name, file_ext = os.path.splitext(img_filename)
        file_ext = '.jpg' # always save as .jpg because for the moment, tensorflow expects jpg
        
        if create_scan:
            document.create_scan()
            scan_file_name = os.path.join(scan_folder_path, img_name + '_' + 'scan' + file_ext)
            if document.scan is None:
                print('Warning: no scan for {}'.format(img_filename) )
            else:
                cv2.imwrite(scan_file_name, document.scan)

        if create_dataset:
            h, w = document.image_data.photo.shape[:2]
            resized_filename = img_name + '_' + str(h) + 'x' + str(w) + file_ext
            cv2.imwrite(os.path.join(dataset_folder_path, resized_filename), document.image_data.photo)
            img_dict = {}
            corners = document.find_corners()
            img_dict['corners'] = corners.reshape(-1).tolist()
            img_dict['transform'] = document.transform.reshape(-1).tolist()[:-1]
            dataset_dict[resized_filename] = img_dict
        
        if move_processed:
            if not os.path.isdir(os.path.join(folder_path, 'processed')):
                os.mkdir(os.path.join(folder_path, 'processed'))
            os.rename(os.path.join(folder_path, img_filename), os.path.join(folder_path, 'processed', img_filename))
    
    dataset_filename = 'dataset.json'
    print('Writing to: ' + dataset_filename)

    with open(os.path.join(dataset_folder_path, dataset_filename), 'w') as file:
        json.dump(dataset_dict, file, indent=4, separators=(',', ': '))

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create scans from a folder of photos')
    parser.add_argument('photo_path', help='The location of a photo or a folder of photos')
    parser.add_argument('document_type_name', help='document type of the photos')
    parser.add_argument('--path_dict_path', help='The location of the path file')
    parser.add_argument('--create_scan', dest='create_scan', action='store_true', help='Create and save scans')
    parser.add_argument('--create_dataset', dest='create_dataset', action='store_true', help='Create dataset from transform and corner data')
    parser.add_argument('--move_processed', dest='move_processed', action='store_true',
                        help='Move images that have been processed to subfolder to avoid having to redo them in case of restart ')
    args = parser.parse_args()
    process(args.photo_path, args.path_dict_path, args.document_type_name, args.create_scan, args.create_dataset, args.move_processed)
import cv2
import os

from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH


def process(src_path: str,  path_dict_path: str, mock_document_type_name: str, debug: bool, move_processed: bool) -> None:
    path_dict_path = path_dict_path or DEFAULT_PATH_DICT_PATH
    if os.path.isdir(src_path):
        img_paths = [os.path.join(src_path, x) for x in os.listdir(src_path) if os.path.splitext(x)[1] in ['.jpg', '.png', '.JPG']]
    else:
        img_paths = [src_path]
    
    scanner = Document_scanner(path_dict_path, log_level='WARNING', mock_document_type_name=mock_document_type_name)
    
    scans_path = os.path.join(os.path.split(src_path)[0], 'scans')
    for document_type_name in scanner.template_df.keys():
        if not os.path.isdir(os.path.join(scans_path, document_type_name)):
            os.makedirs(os.path.join(scans_path, document_type_name))

    for img_path in img_paths:
        print(img_path)
        document = scanner.develop_document(img_path)
        
        folder_path, img_name = os.path.split(img_path)
        file_stem, file_ext = os.path.splitext(img_name)
        file_ext = '.jpg' # always save as .jpg because for the moment, tensorflow expects jpg
        
        filename = os.path.join(scans_path, document.document_type_name, file_stem + '_' + 'scan' + file_ext)
        if debug:
            print(filename)
        if document.scan is None:
            return 
        cv2.imwrite(filename, document.scan)
        
        if move_processed:
            if not os.path.isdir(os.path.join(folder_path, 'processed')):
                os.mkdir(os.path.join(folder_path, 'processed'))
            os.rename(img_path, os.path.join(folder_path, 'processed', img_name))
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create scans from a folder of photos')
    parser.add_argument('photo_path', help='The location of a photo or a folder of photos')
    parser.add_argument('--path_dict_path', help='The location of the path file')
    parser.add_argument('--mock_document_type_name', default=None, help='mock_document_type_name to use')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Display pictures and values')
    parser.add_argument('--move_processed', dest='move_processed', action='store_true',
                        help='Move images that have been processed to subfolder to avoid having to redo them in case of restart ')
    args = parser.parse_args()
    process(args.photo_path, args.path_dict_path, args.mock_document_type_name, args.debug, args.move_processed)
import cv2
import os

from document_scanner.cv_wrapper import crop_sections
from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH

def process(src_path: str, path_dict_path: str, mock_document_type_name: str, debug: bool,
            move_processed: bool) -> None:
    path_dict_path = path_dict_path or DEFAULT_PATH_DICT_PATH
    if os.path.isdir(src_path):
        img_paths = [os.path.join(src_path, x) for x in os.listdir(src_path) if
                     os.path.splitext(x)[1] in ['.jpg', '.png', '.JPG']]
    else:
        img_paths = [src_path]

    scanner = Document_scanner(path_dict_path, log_level='WARNING', mock_document_type_name=mock_document_type_name)

    crops_path = os.path.join(os.path.split(src_path)[0], 'crops')
    for ind in scanner.model_df.index:
        if not os.path.isdir(os.path.join(crops_path, *ind)):
            os.makedirs(os.path.join(crops_path, *ind))

    for img_path in img_paths:
        print(img_path)

        folder_path, img_name = os.path.split(img_path)
        file_stem, file_ext = os.path.splitext(img_name)
        file_ext = '.jpg'  # always save as .jpg because for the moment, tensorflow expects jpg
        file_stem = file_stem[:-len('_scan')] if file_stem.endswith('_scan') else file_stem
        
        scan = cv2.imread(img_path, 0)
        field_df = crop_sections(scan, scanner.field_data_df.xs(mock_document_type_name))
        #field_df = field_df.merge(scanner.field_data_df.xs(mock_document_type_name), left_index = True, right_index = True)

        for box_name, row in field_df.iterrows():
            #print(row)
            filename = os.path.join(crops_path, mock_document_type_name, row['model_name'], file_stem + '_' + box_name + file_ext)
            if debug:
                print(filename)
            cv2.imwrite(filename, row['crop'])

        if move_processed:
            if not os.path.isdir(os.path.join(folder_path, 'processed')):
                os.mkdir(os.path.join(folder_path, 'processed'))
            os.rename(img_path, os.path.join(folder_path, 'processed', img_name))
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Crop and store boxes from a folder of images')
    parser.add_argument('scan_path', help='The location of a scan or a folder of scans')
    parser.add_argument('--path_dict_path', help='The location of the path file')
    parser.add_argument('--mock_document_type_name', help='mock_document_type_name to use')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Display pictures and values')
    parser.add_argument('--move_processed', dest='move_processed', action='store_true',
                        help='Move images that have been processed to subfolder to avoid having to redo them in case of restart ')
    args = parser.parse_args()
    process(args.scan_path, args.path_dict_path, args.mock_document_type_name, args.debug, args.move_processed)
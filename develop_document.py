from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH


def develop_document(image_path: str, path_dict_path: str, debug: bool) -> None:
    path_dict_path = path_dict_path or DEFAULT_PATH_DICT_PATH
    scanner = Document_scanner(path_dict_path)
    scanner.develop_document(image_path, debug)
    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Read out form from photo')
    parser.add_argument('image_path', help='The location of the image file')
    parser.add_argument('--path_dict_path', help='The location of the path dict')
    parser.add_argument('--mock_document_type_name', help='document_type_name to use')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Display pictures and values')
    args = parser.parse_args()
    develop_document(args.image_path, args.path_dict_path, args.debug)
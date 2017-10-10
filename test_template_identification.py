from document_scanner.document import Document
from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import DEFAULT_PATH_DICT_PATH


def identify_template(image_path: str, mock_document_type_name: str, path_dict_path: str, debug: bool) -> None:
    path_dict_path = path_dict_path or DEFAULT_PATH_DICT_PATH
    scanner = Document_scanner.for_document_identification(path_dict_path, mock_document_type_name)

    document = Document.from_path(image_path, scanner.business_logic_class)
    #document.document_type_name = mock_document_type_name
    document.find_match(scanner.template_df.loc[mock_document_type_name, 'template'], scanner.orb)
    if debug:
        document.print_template_match_quality()
    if not document.can_create_scan():
        if debug:
            document.show_match_with_template()
        document.error_reason = 'image_quality'
        print('Identified insufficient points for template matching; aborting')
        return
    print('Identified points for template matching')
    document.create_scan()
    print('Created scan from original photo')
    if debug:
        document.show_match_with_template()
        document.show_scan()
    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Read out form from photo')
    parser.add_argument('image_path', help='The location of the image file')
    parser.add_argument('mock_document_type_name', help='document_type_name to use')
    parser.add_argument('--path_dict_path', help='The location of the path dict')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Display pictures and values')
    args = parser.parse_args()
    identify_template(args.image_path, args.mock_document_type_name, args.path_dict_path, args.debug)
import document_scanner.document_scanner as ds


def develop_document(config_path: str, image_path: str, debug: bool) -> None:
    scanner = ds.Document_scanner(config_path)
    scanner.develop_document(image_path, debug)
    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Read out form from photo')
    parser.add_argument('config_path', help='The location of the config file')
    parser.add_argument('image_path', help='The location of the image file')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Display pictures and values')
    args = parser.parse_args()
    develop_document(args.config_path, args.image_path, args.debug)
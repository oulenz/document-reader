import document_scanner as ds
import cv2
import os


def process(config_path: str, image_path: str, debug: bool) -> None:
    config = ds.get_config(config_path, image_path)
    if debug:
        print(config)

    template = cv2.imread(config['template_path'], 0)

    if os.path.isdir(config['image_path']):
        img_paths = [os.path.join(config['image_path'], x) for x in os.listdir(config['image_path']) if os.path.splitext(x)[1] in ['.jpg', '.png']]
    else:
        img_paths = [config['image_path']]
    # print(len(img_paths))
    for img_path in img_paths:
        print(img_path)
        scan = ds.find_document(template, img_path, debug)
        ds.read_document(scan, template, img_path, config, debug)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Read out form from photo')
    parser.add_argument('config_path', help='The location of the config file')
    parser.add_argument('image_path', help='The location of the image file (overrides path in config file)', nargs='?')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Display pictures and values')
    args = parser.parse_args()
    process(args.config_path, args.image_path, args.debug)
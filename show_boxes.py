import cv2

from document_scanner.document_scanner import Document_scanner


def show_boxes(path_dict_path: str, img_path: str) -> None:
    path_dict = Document_scanner.parse_path_dict(path_dict_path)
    template = cv2.imread(path_dict['template_path'], 0)
    image = cv2.imread(img_path, 0)

    boxes = ds.parse_boxes(config)
    ds.show_boxes(image, boxes)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Crop and store boxes from a folder of images')
    parser.add_argument('config_path', help='The location of the config file')
    parser.add_argument('image_path', help='The location of an image file, default is the template', nargs='?')
    args = parser.parse_args()
    show_boxes(args.config_path, args.image_path)
import cv2
import document_scanner as ds


def show_boxes(config_path: str, img_path: str) -> None:
    config = ds.parse_config(config_path)
    template = cv2.imread(config['template_path'], 0)

    if img_path is not None:
        photo = cv2.imread(img_path, 0)
        image = ds.find_document(template, photo, debug = False)
    else:
        image = template

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
import cv2
import document_scanner as ds


def show_boxes(config_path: str) -> None:
    config = ds.parse_config(config_path)
    boxes = ds.parse_boxes(config)
    template = cv2.imread(config['template_path'], 0)
    ds.show_boxes(template, boxes)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Crop and store boxes from a folder of images')
    parser.add_argument('config_path', help='The location of the config file')
    args = parser.parse_args()
    show_boxes(args.config_path)
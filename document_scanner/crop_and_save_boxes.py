import cv2
import document_scanner as ds
import os


def process(config_path: str, image_path: str, debug: bool) -> None:
    config = ds.parse_config(config_path, image_path)
    # overrule output folder:
    config['output_path'] = os.path.join(os.path.split(image_path)[0], 'crops')
    if debug:
        print(config)

    template = cv2.imread(config['template_path'], 0)

    if os.path.isdir(config['image_path']):
        img_paths = [os.path.join(config['image_path'], x) for x in os.listdir(config['image_path']) if os.path.splitext(x)[1] in ['.jpg', '.png', '.JPG']]
    else:
        img_paths = [config['image_path']]
    boxes = ds.parse_boxes(config)
    for img_path in img_paths:
        print(img_path)
        scan = ds.find_document(template, img_path, debug)
        ds.crop_boxes(scan, boxes, img_path, config)
        folder_path, img_name = os.path.split(img_path)
        # move images that are done to subfolder to avoid having to redo them in case of restart
        if not os.path.isdir(os.path.join(folder_path, 'processed')):
            os.mkdir(os.path.join(folder_path, 'processed'))
        os.rename(img_path, os.path.join(folder_path, 'processed', img_name))
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Crop and store boxes from a folder of images')
    parser.add_argument('config_path', help='The location of the config file')
    parser.add_argument('image_path', help='The location of an image file or a folder of image files')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Display pictures and values')
    args = parser.parse_args()
    process(args.config_path, args.image_path, args.debug)
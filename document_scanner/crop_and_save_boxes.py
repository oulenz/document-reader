import cv2
import document_scanner as ds
import os


def process(config_path: str, src_path: str, debug: bool, move_processed: bool) -> None:
    if os.path.isdir(src_path):
        img_paths = [os.path.join(src_path, x) for x in os.listdir(src_path) if os.path.splitext(x)[1] in ['.jpg', '.png', '.JPG']]
    else:
        img_paths = [src_path]
    config = ds.parse_config(config_path)
    if debug:
        print(config)
    output_path = os.path.join(os.path.split(src_path)[0], 'crops')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    template = cv2.imread(config['template_path'], 0)
    boxes = ds.parse_boxes(config)
    for img_path in img_paths:
        print(img_path)
        photo = cv2.imread(img_path, 0)
        scan = ds.find_document(template, photo, debug)
        box_selection = ds.crop_boxes(scan, boxes)
        folder_path, img_name = os.path.split(img_path)
        file_stem, file_ext = os.path.splitext(img_name)
        file_ext = '.jpg' # always save as .jpg because for the moment, tensorflow expects jpg
        for box_name, crop in box_selection['crop'].iteritems():
            filename = os.path.join(output_path, file_stem + '_' + box_name + file_ext)
            print(filename)
            cv2.imwrite(filename, crop)
        if move_processed:
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
    parser.add_argument('--move_processed', dest='move_processed', action='store_true',
                        help='Move images that have been processed to subfolder to avoid having to redo them in case of restart ')
    args = parser.parse_args()
    process(args.config_path, args.image_path, args.debug, args.move_processed)
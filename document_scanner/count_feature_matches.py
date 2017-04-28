import cv2
import document_scanner as ds
import os
import pandas as pd


def process(config_path: str, src_path: str) -> None:
    config = ds.parse_config(config_path)
    template = cv2.imread(config['template_path'], 0)

    if not os.path.isdir(src_path):
        photo = cv2.imread(src_path, 0)
        _, _, matches = ds.get_matching_points(template, photo, False)
        print(len(matches))
    else:
        feature_match_counts = []

        for label in [x for x in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, x))]:
            label_path = os.path.join(src_path, label)
            for img_name in [x for x in os.listdir(label_path) if os.path.splitext(x)[1] in ['.jpg', '.png', '.JPG']]:
                img_path = os.path.join(label_path, img_name)
                print(img_path)
                photo = cv2.imread(img_path, 0)
                _, _, matches = ds.get_matching_points(template, photo, False)
                feature_match_counts.append({'label': label, 'img_name': img_name, 'matches': len(matches)})

        pd.DataFrame(feature_match_counts).to_csv(os.path.join(src_path, 'feature_match_counts.csv'), sep='|', index=False)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Match template to a folder of images, store number of features matched')
    parser.add_argument('config_path', help='The location of the config file')
    parser.add_argument('image_path', help='Folder containing subfolders with image files')
    args = parser.parse_args()
    process(args.config_path, args.image_path)
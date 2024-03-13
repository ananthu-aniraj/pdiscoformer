import pandas as pd
import copy
import os
import argparse
from utils.data_utils.dataset_utils import load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare PartImagenet OOD dataset'
    )
    parser.add_argument('--anno_path', type=str, required=True)
    parser.add_argument('--train_test_split_file', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    return parser.parse_args()


def prepare_pimagenet_ood(args):
    coco_json_path = args.anno_path
    train_test_split_path = args.train_test_split_file
    data = load_json(coco_json_path)

    columns = ["image_id", "is_test", "label_id", "label_name", "image_name"]

    train_test_csv = pd.read_csv(train_test_split_path, sep="\t", header=None, names=columns)
    image_id_to_test = {}
    for index, row in train_test_csv.iterrows():
        image_id_to_test[row["image_id"]] = row["is_test"]

    train_data = []
    test_data = []
    for image in data["images"]:
        label_name = image["file_name"].split("_")[0]
        image["file_name"] = os.path.join(label_name, image["file_name"])
        if image_id_to_test[image["id"]]:
            test_data.append(image)
        else:
            train_data.append(image)

    train_annotations = []
    test_annotations = []
    for ann in data["annotations"]:
        if image_id_to_test[ann["image_id"]]:
            test_annotations.append(ann)
        else:
            train_annotations.append(ann)

    # Now adjust the image ids in the annotations
    train_img_count = 0
    original_img_id_to_new_img_id = {}
    for image in train_data:
        original_img_id = image["id"]
        image["id"] = train_img_count
        original_img_id_to_new_img_id[original_img_id] = train_img_count
        train_img_count += 1

    for ann in train_annotations:
        ann["image_id"] = original_img_id_to_new_img_id[ann["image_id"]]

    test_img_count = 0
    original_img_id_to_new_img_id_test = {}
    for image in test_data:
        original_img_id = image["id"]
        image["id"] = test_img_count
        original_img_id_to_new_img_id_test[original_img_id] = test_img_count
        test_img_count += 1

    for ann in test_annotations:
        ann["image_id"] = original_img_id_to_new_img_id_test[ann["image_id"]]

    # Save the new json files
    train_json = copy.deepcopy(data)
    test_json = copy.deepcopy(data)
    train_json["images"] = train_data
    train_json["annotations"] = train_annotations
    test_json["images"] = test_data
    test_json["annotations"] = test_annotations
    save_json(train_json, os.path.join(args.output_path, "train_train.json"))
    save_json(test_json, os.path.join(args.output_path, "train_test.json"))


if __name__ == '__main__':
    arguments = parse_args()
    prepare_pimagenet_ood(arguments)

import argparse
import os
import shutil
import sys
from datetime import datetime
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchvision.datasets import CocoDetection
from yaml.loader import SafeLoader


# matplotlib.use('Agg')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="configs/clip_gen.yaml")
    args = parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config




def set_logging(cfg):
    now = datetime.now()
    date_save_string = now.strftime("%d%m%Y_%H%M")
    checkpoint_dir = os.path.join(
        cfg["logging"]["root_dir"],
        cfg["logging"]["checkpoint_dir"],
        date_save_string,
    )
    print(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, "log.log")
    return log_file, checkpoint_dir


def get_groundtruths(dataset, show_image=False):
    gt = np.empty((0, 7))
    for index in range(len(dataset)):
        image_id = dataset.ids[index]
        image, anns = dataset[index]
        image = np.array(image)
        bounding_box_list = []
        class_list = []
        for ann in anns:
            bounding_box_list.append(ann['bbox'])
            class_list.append(ann['category_id'])

        if (show_image):
            bbox = bounding_box_list[0]
            bbox = [int(x) for x in bbox]
            print(bbox)
            image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
            plt.imshow(image)
            plt.show()
            break
        image_id = np.array(image_id)
        bounding_box_list = np.array(bounding_box_list)
        image_id_list = np.ones((len(class_list), 1)) * image_id
        scores_list = np.ones((len(class_list), 1))
        class_list = np.array(class_list).reshape((len(class_list), 1))
        # ["image_id", "bbox_y", "bbox_x", "w", "h", "score", "class_label"]
        if (len(bounding_box_list != 0)):
            gt_idx = np.hstack((image_id_list, bounding_box_list, scores_list, class_list))
            gt = np.vstack((gt, gt_idx))
    return gt

def get_clip_embedding(dataset,class_id,class_name):
    gt = np.empty((0, 7))
    for index in range(len(dataset)):
        image_id = dataset.ids[index]
        image, anns = dataset[index]
        image = np.array(image)
        bounding_box_list = []
        class_list = []
        for ann in anns:
            bounding_box_list.append(ann['bbox'])
            class_list.append(ann['category_id'])

        if (show_image):
            bbox = bounding_box_list[0]
            bbox = [int(x) for x in bbox]
            print(bbox)
            image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
            plt.imshow(image)
            plt.show()
            break
        image_id = np.array(image_id)
        bounding_box_list = np.array(bounding_box_list)
        image_id_list = np.ones((len(class_list), 1)) * image_id
        scores_list = np.ones((len(class_list), 1))
        class_list = np.array(class_list).reshape((len(class_list), 1))
        # ["image_id", "bbox_y", "bbox_x", "w", "h", "score", "class_label"]
        if (len(bounding_box_list != 0)):
            gt_idx = np.hstack((image_id_list, bounding_box_list, scores_list, class_list))
            gt = np.vstack((gt, gt_idx))
    return gt

def main(cfg):
    dataset_root = cfg["data"]["root"]
    dataset = CocoDetection(root=os.path.join(dataset_root, "data"),
                            annFile=os.path.join(dataset_root, "labels.json"))
    class_name_list=[]
    class_id_list=[]
    for index in range(len(dataset.coco.cats)):
        print(index)
        cat = dataset.coco.cats[index]
        class_id_list.append(cat["id"])
        class_name_list.append(cat["name"])


    gt = get_groundtruths(dataset)
    print(gt)

if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.c)

    sys.exit(main(cfg))

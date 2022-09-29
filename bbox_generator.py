import argparse
import copy
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
from tqdm import tqdm

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



def get_clip_embedding(dataset,class_id,class_name):
    root = os.path.join("/home/psrahul/MasterThesis/datasets/BBoxGroundtruths/PASCAL_3_2/train/",class_name)
    os.makedirs(root,exist_ok=True)
    counter=0
    for index in tqdm(range(len(dataset))):
        image_id = dataset.ids[index]
        image, anns = dataset[index]
        image = np.array(image)
        bounding_box_list = []
        class_list = []
        for ann in anns:
            bbox=ann['bbox']
            category_id=ann['category_id']
            if(category_id==class_id) and (ann['difficult']==0):
                bbox = [int(x) for x in bbox]
                image_cropped=copy.deepcopy(image)
                image_cropped = image_cropped[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
                plt.imsave(os.path.join(root,str(counter)+".png"),image_cropped)
                counter+=1
                plt.close("all")

def main(cfg):
    dataset_root = cfg["data"]["root"]
    dataset = CocoDetection(root=os.path.join(dataset_root, "data"),
                            annFile=os.path.join(dataset_root, "labels.json"))
    class_name_list=[]
    class_id_list=[]
    for index in tqdm(range(len(dataset.coco.cats))):
        cat = dataset.coco.cats[index]
        class_id_list.append(cat["id"])
        class_name_list.append(cat["name"])

    for index in tqdm(range(len(class_id_list))):
        get_clip_embedding(dataset,class_id_list[index],class_name_list[index])


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.c)

    sys.exit(main(cfg))

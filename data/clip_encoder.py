import os
import sys

import clip
import torch
from PIL import Image
from tqdm import tqdm
image_root = "/home/psrahul/MasterThesis/datasets/PASCAL_2012_ZETA_CA/support_images/base_classes/val/"
import numpy as np

image_list = ["aeroplane.png",
              "dog.png",
              "sheep.png"]


def main():
    data_root="/home/psrahul/MasterThesis/datasets/BBoxGroundtruths/PASCAL_3_2/train/"
    class_names=sorted(os.listdir(data_root))
    with torch.no_grad():
        clip_model, clip_preprocess = clip.load("ViT-B/16", device="cuda")
        clip_model = clip_model.eval()
    for class_name in tqdm(class_names):
        image_list=sorted(os.listdir(os.path.join(data_root,class_name)))
        clip_embeddings = np.zeros((len(image_list), 512))
        counter=0
        for image in tqdm(image_list):
            image_path=os.path.join(data_root,class_name,image)

            with torch.no_grad():
                image = Image.open(os.path.join(image_root, image_path))
                image = clip_preprocess(image).unsqueeze(0)
                image_clip_embedding = clip_model.encode_image(image.cuda())
                image_clip_embedding = image_clip_embedding.cpu().numpy()
                clip_embeddings[counter] = image_clip_embedding
                counter+=1

        clip_embeddings=np.mean(clip_embeddings,axis=0)

        np.save(os.path.join("/home/psrahul/MasterThesis/datasets/BBoxGroundtruths/PASCAL_3_2/train/",class_name+".npy"), clip_embeddings)


if __name__ == "__main__":
    sys.exit(main())

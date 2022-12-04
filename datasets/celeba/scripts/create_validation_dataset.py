import os
import sys
import shutil
from utils_segmentation import find_corresponding_images
import random


def create_validation_images(input_images_dir, input_masks_dir, output_dir, output_masks_dir):
    """
    Copies the subset of the images from input_images_dir which contain segmenentations in masks_dir to output_dir.
    :param masks_dir: path to the folder which contains the segmentation masks
    :param input_images_dir: path to the folder which contains the input images
    :param output_dir: path to the folder where to copy the input images
    :param input_ext: the extension of the input images (by default png)
    :return None
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if not os.path.exists(output_masks_dir):
        os.mkdir(output_masks_dir)
    
    img_paths = []
    for img in os.listdir(input_images_dir):
        img_paths.append(os.path.join(input_images_dir, img))
    random_imgs = random.sample(img_paths, k=int(len(img_paths)/10))
 
    for idx, img_path in enumerate(random_imgs):
        basename = os.path.basename(img_path)
        mask_name = basename.replace('.png', '.bmp')
        shutil.move(img_path, output_dir)
        shutil.move(os.path.join(input_masks_dir, mask_name), output_masks_dir)

if __name__ == '__main__':
    base_dir = "/Users/shubhamdawande/my-projects/hair_segmentation/datasets/celeba/"
    i_dir = base_dir + "train_imgs"
    i_masks_dir = base_dir + "train_masks"
    o_dir = base_dir + "validation_imgs"
    o_masks_dir = base_dir + "validation_masks"

    create_validation_images(i_dir, i_masks_dir, o_dir, o_masks_dir)


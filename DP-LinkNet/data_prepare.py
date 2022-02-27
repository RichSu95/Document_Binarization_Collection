# Prepare dataset for training

import os

import cv2

from utils import get_patches

TILE_SIZE = 256
print("Image patch size:", TILE_SIZE, "x", TILE_SIZE)

data_root_image = "./dataset"
data_root_mask = "./data_GT"
data_save = "./data"
img_list = os.listdir(data_root_image)
img_list.sort()

data_train_dir = os.path.join(data_save, "train")
if not os.path.exists(data_train_dir):
    os.makedirs(data_train_dir)

# img_patches, msk_patches = [], []  # patches for each image or ground truth
total_img_patches, total_msk_patches = [], []  # patches for all the images or ground truths

for idx in range(len(img_list)):
    if os.path.isdir(os.path.join(data_root_image, img_list[idx])):
        continue

    print("Now processing image:", os.path.join(data_root_image, img_list[idx]))
    (fname, fext) = os.path.splitext(img_list[idx])
    img = cv2.imread(os.path.join(data_root_image, img_list[idx]))
    msk = cv2.imread(os.path.join(data_root_mask, fname + ".png"))

    # extract the patches from the original document images and the corresponding ground truths
    img_patch_locations, img_patches = get_patches(img, TILE_SIZE, TILE_SIZE)
    msk_patch_locations, msk_patches = get_patches(msk, TILE_SIZE, TILE_SIZE)

    print("Patches extracted:", len(img_patches))
    for idy in range(len(img_patches)):
        total_img_patches.append(img_patches[idy])
        total_msk_patches.append(msk_patches[idy])

print("Total number of train patches:", len(total_img_patches))
for idz in range(len(total_img_patches)):
    cv2.imwrite(os.path.join(data_train_dir, str(idz) + "_img.png"), total_img_patches[idz])
    cv2.imwrite(os.path.join(data_train_dir, str(idz) + "_mask.png"), total_msk_patches[idz])

print("Done")

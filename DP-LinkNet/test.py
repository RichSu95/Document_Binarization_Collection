import os
from time import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable as V

# from networks.unet import UNet
# from networks.dunet import DUNet
from networks.dplinknet import LinkNet34, DLinkNet34, DPLinkNet34
from utils import get_patches, stitch_together

BATCHSIZE_PER_CARD = 32


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


TILE_SIZE = 256
DATA_NAME = "DIBCO"  # BickleyDiary, DIBCO, PLM
DEEP_NETWORK_NAME = "DPLinkNet34"  # LinkNet34, DLinkNet34, DPLinkNet34

img_indir = "test_set/image_test2019/"
print("Image input directory:", img_indir)

img_outdir = os.path.join(img_indir, "Binarized")
if not os.path.exists(img_outdir):
    os.makedirs(img_outdir)
print("Image output directory:", img_outdir)

img_list = os.listdir(img_indir)
img_list.sort()

if DEEP_NETWORK_NAME == "DPLinkNet34":
    solver = TTAFrame(DPLinkNet34)
elif DEEP_NETWORK_NAME == "DLinkNet34":
    solver = TTAFrame(DLinkNet34)
elif DEEP_NETWORK_NAME == "LinkNet34":
    solver = TTAFrame(LinkNet34)
else:
    print("Deep network not found, please have a check!")
    exit(0)
# print(solver.net)
# summary(solver.net, input_size=(3, TILE_SIZE, TILE_SIZE))  # summary(your_model, input_size=(channels, H, W))

print("Now loading the model weights:", "weights/" + DATA_NAME.lower() + "_" + DEEP_NETWORK_NAME.lower() + ".th")
solver.load("weights/" + DATA_NAME.lower() + "_" + DEEP_NETWORK_NAME.lower() + ".th")

start_time = time()
for idx in range(len(img_list)):
    if os.path.isdir(os.path.join(img_indir, img_list[idx])):
        continue

    print("Now processing image:", img_list[idx])
    fname, fext = os.path.splitext(img_list[idx])
    img_input = os.path.join(img_indir, img_list[idx])
    img_output = os.path.join(img_outdir, fname +".png")

    img = cv2.imread(img_input)
    locations, patches = get_patches(img, TILE_SIZE, TILE_SIZE)
    masks = []
    for idy in range(len(patches)):
        msk = solver.test_one_img_from_path(patches[idy])
        masks.append(msk)
    prediction = stitch_together(locations, masks, tuple(img.shape[0:2]), TILE_SIZE, TILE_SIZE)
    prediction[prediction >= 5.0] = 255
    prediction[prediction < 5.0] = 0
    # prediction = np.concatenate([prediction[:, :, None], prediction[:, :, None], prediction[:, :, None]], axis=2)
    cv2.imwrite(img_output, prediction.astype(np.uint8))

print("Total running time: %f sec." % (time() - start_time))
print("Finished!")

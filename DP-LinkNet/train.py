import os
from time import time

import torch
import torch.utils.data as data

from data import ImageFolder
from framework import MyFrame
from loss import dice_bce_loss

# from networks.unet import UNet
# from networks.dunet import DUNet
from networks.dplinknet import LinkNet34, DLinkNet34, DPLinkNet34

SHAPE = (256, 256)
DATA_NAME = "DIBCO"  # BickleyDiary, DIBCO, PLM
DEEP_NETWORK_NAME = "DPLinkNet34"
print("Now training dataset: {}, using network model: {}".format(DATA_NAME, DEEP_NETWORK_NAME))

train_root = "./data/train/"
imagelist = list(filter(lambda x: x.find("img") != -1, os.listdir(train_root)))
trainlist = list(map(lambda x: x[:-8], imagelist))
log_name = DATA_NAME.lower() + "_" + DEEP_NETWORK_NAME.lower()

BATCHSIZE_PER_CARD = 32

if DEEP_NETWORK_NAME == "DPLinkNet34":
    solver = MyFrame(DPLinkNet34, dice_bce_loss, 2e-4)
elif DEEP_NETWORK_NAME == "DLinkNet34":
    solver = MyFrame(DLinkNet34, dice_bce_loss, 2e-4)
elif DEEP_NETWORK_NAME == "LinkNet34":
    solver = MyFrame(LinkNet34, dice_bce_loss, 2e-4)
else:
    print("Deep network not found, please have a check!")
    exit(0)

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, train_root)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4)

mylog = open("logs/" + log_name + ".log", "w")
no_optim = 0
total_epoch = 500
train_epoch_best_loss = 100.

tic = time()
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print("********", file=mylog)
    print("epoch:", epoch, "    time:", int(time() - tic), file=mylog)
    print("train_loss:", train_epoch_loss, file=mylog)
    print("SHAPE:", SHAPE, file=mylog)
    print("********")
    print("epoch:", epoch, "    time:", int(time() - tic))
    print("train_loss:", train_epoch_loss)
    print("SHAPE:", SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save("weights/" + log_name + ".th")

    if no_optim > 20:
        print("early stop at %d epoch" % epoch, file=mylog)
        print("early stop at %d epoch" % epoch)
        break

    if no_optim > 10:
        if solver.old_lr < 1e-7:
            break
        solver.load("weights/" + log_name + ".th")
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()

print("Finish!", file=mylog)
print("Finish!")
mylog.close()

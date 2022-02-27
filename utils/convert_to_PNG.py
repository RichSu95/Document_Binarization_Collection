
from PIL import Image
import numpy as np
import os
from skimage.transform import rotate

read_dir = "D:\\Research_Project\\DIBCOdataset\\DIBCO2019\\degraded\\"
save_dir = "D:\\Research_Project\\DeepOtsu\\image_test2019\\"

files = os.listdir(read_dir)

count = 0
for ims in range(len(files)):
    print(str(files[ims]))
    im = Image.open(read_dir+str(files[ims]))
    imarr = np.array(im)

    arr2im = Image.fromarray(imarr)
    arr2im.save(save_dir+str(count)+".png")
    cnt = cnt+1
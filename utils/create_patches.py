from sklearn.feature_extraction.image import extract_patches_2d
from PIL import Image
import numpy as np
import os
from skimage.transform import rotate

read_dir = "D:\\Research_Project\\DIBCOdataset\\PHIBD2012\\GT\\"
save_dir = "Data\\"

files = os.listdir(read_dir)

count = 0
for ims in range(len(files)):
    print(str(files[ims]))
    im = Image.open(read_dir+str(files[ims]))
    imarr = np.array(im)

    patches = extract_patches_2d(imarr, patch_size= (256,256), max_patches=10, random_state=1)
    patlr = []
    patud = []

    for j1 in range(0, len(patches)):
        temp = patches[j1]
        temp = np.fliplr(temp)
        patlr.append(temp.astype('uint8'))

    for j2 in range(0, len(patches)):
        temp = patches[j2]
        temp = np.flipud(temp)
        patud.append(temp.astype('uint8'))


    patches = list(patches)
    patches.extend(patlr)
    patches.extend(patud)

    for i in range(0,len(patches)):
        pat = patches[i]
        arr2im = Image.fromarray(pat)
        arr2im.save(save_dir+str(count)+".png")
        cnt = cnt+1


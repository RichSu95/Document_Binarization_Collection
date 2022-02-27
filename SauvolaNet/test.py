import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
print(tf.__version__)
from matplotlib import pyplot
import sys
import pandas as pd
from dataUtils import collect_binarization_by_dataset, DataGenerator
from testUtils import prepare_inference, find_best_model
from layerUtils import *
from metrics import *
from PIL import Image

model_root = 'pretrained_models/'
for this in os.listdir(model_root):
    if this.endswith('.h5'):
        model_filepath = os.path.join(model_root, this)
        model = prepare_inference(model_filepath)
model.summary()

dataset_lut = collect_binarization_by_dataset('test_set')
test_datasets = ["DIBCO2019"]
for this in test_datasets :
    all_metrics = []
    test_datagen = DataGenerator(dataset_lut[this], output_shape=None, mode='testing')
    L = len(test_datagen)
    for i in range(L) :
        x, y = test_datagen[i]
        print(x.shape)
        z = model.predict(x)
        print(z.shape)
        #mycode start
        z_copy = np.copy(z)
        xx = np.reshape(z_copy, (z_copy.shape[1],z_copy.shape[2]))
        print(xx)
        temp = np.ones((xx.shape[0], xx.shape[1]))*255
        temp[xx==255] = 0
        zz = Image.fromarray((xx).astype(np.uint8))
        zz.save("outputs//"+str(i)+".png")
        #mycode ends
        pyplot.figure(figsize=(15,5))
        pyplot.subplot(131)
        pyplot.imshow(x[0,...,0], cmap='gray')
        pyplot.title('{}-{}'.format(x.min(), x.max()))
        pyplot.subplot(132)
        pyplot.imshow(y[0,...,0], cmap='gray', vmin=-1, vmax=1)
        pyplot.title('{}-{}'.format(y.min(), y.max()))
        pyplot.subplot(133)
        pyplot.imshow(z[0,...,0]>0, cmap='gray', vmin=0, vmax=1)
        pyplot.title(f'predicted {z.min():.2f} - {z.max():.2f}')
        f1 = Fmeasure(z[0,...,0]>0,y[0,...,0]>0)
        psnr = Psnr(z[0,...,0]>0,y[0,...,0]>0)
        Pf = Pfmeasure(z[0,...,0]>0,y[0,...,0]>0)
        drd = DRD(z[0,...,0]>0,y[0,...,0]>0)
        all_metrics.append([f1, psnr, Pf, drd])
        pyplot.title(f'{i} :f1={f1*100:.2f}% | PSNR={psnr:.2f}dB | Pf={Pf*100:.2f}% | Drd={drd:.2f}%')
        pyplot.show()
        pyplot.savefig('results.png')

    print('-' * 100)
    print("Total average : " +this+" "+ str(np.mean(all_metrics,axis=0)))
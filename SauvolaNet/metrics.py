# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:02:46 2019

@author: VIPlab
"""
import numpy as np
import cv2
import math
from scipy import ndimage as ndi

#predict_img = 'E:\Document-Binarization\DIBCO_metrics\DIBCO_metrics\P03_adotsu.tif'
#predict_img = cv2.imread(predict_img, 0)
#GT_img = 'E:\Document-Binarization\DIBCO_metrics\DIBCO_metrics\P03_GT.tif'
#GT_img = cv2.imread(GT_img, 0)
#predict_img_ = np.copy(predict_img)
#predict_img_ = predict_img_/255
#GT_img_ = np.copy(GT_img)
#GT_img_ = GT_img_/255


G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 0, 0], dtype=np.bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], dtype=np.bool)

def bwmorph_thin(image, n_iter=None):
    # check parameters
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter
    
    # check that we have a 2d binary image, and convert it
    # to uint8
    skel = np.array(image).astype(np.uint8)
    
    if skel.ndim != 2:
        raise ValueError('2D array required')
    if not np.all(np.in1d(image.flat,(0,1))):
        raise ValueError('Image contains values other than 0 and 1')

    # neighborhood mask
    mask = np.array([[ 8,  4,  2],
                     [16,  0,  1],
                     [32, 64,128]],dtype=np.uint8)

    # iterate either 1) indefinitely or 2) up to iteration limit
    while n != 0:
        before = np.sum(skel) # count points before thinning
        
        # for each subiteration
        for lut in [G123_LUT, G123P_LUT]:
            # correlate image with neighborhood mask
            N = ndi.correlate(skel, mask, mode='constant')
            # take deletion decision from this subiteration's LUT
            D = np.take(lut, N)
            # perform deletion
            skel[D] = 0
            
        after = np.sum(skel) # coint points after thinning
        
        if before == after:
            # iteration had no effect: finish
            break
            
        # count down to iteration limit (or endlessly negative)
        n -= 1
    
    return skel.astype(np.bool)


def Fmeasure(predict_img_,GT_img_):
    temp_tp = (1-predict_img_) * (1-GT_img_)
    temp_fp = (1-predict_img_) * GT_img_
    temp_fn = predict_img_ * (1-GT_img_)
    temp_tn = predict_img_ * GT_img_
    count_tp = sum(sum(temp_tp))
    count_fp = sum(sum(temp_fp))
    count_fn = sum(sum(temp_fn))
    count_tn = sum(sum(temp_tn))
    temp_p = count_tp / (count_fp + count_tp + 1e-4)
    temp_r = count_tp / (count_fn + count_tp + 1e-4)
    temp_f = 2 * (temp_p * temp_r) / (temp_p + temp_r + 1e-4)
    return temp_f

def Psnr(predict_img_,GT_img_):
    temp_fp = (1-predict_img_) * GT_img_
    temp_fn = predict_img_ * (1-GT_img_)
    xm = GT_img_.shape[0]
    ym = GT_img_.shape[1]
    fp_fn = temp_fp + temp_fn
    fp_fn[fp_fn>0] = 1
    fp_fn[fp_fn==0] = 0
    err=sum(sum(fp_fn)) / (xm * ym) 
    temp_PSNR = 10 * math.log( 1 / err,10)
    return temp_PSNR

def Pfmeasure(predict_img_,GT_img_):
    N_GT_img_ = 1 - GT_img_
    skel_GT = bwmorph_thin(N_GT_img_)
    skel_GT = (skel_GT).astype('uint8')
    skel_GT = 1 - skel_GT
    temp_tp = (1-predict_img_) * (1-GT_img_)
    temp_fp = (1-predict_img_) * GT_img_
    temp_fn = predict_img_ * (1-GT_img_)
    temp_tn = predict_img_ * GT_img_
    count_tp = sum(sum(temp_tp))
    count_fp = sum(sum(temp_fp))
    count_fn = sum(sum(temp_fn))
    count_tn = sum(sum(temp_tn))
    temp_p = count_tp / (count_fp + count_tp + 1e-4) 
    temp_skl_tp = (1-predict_img_) * (1-skel_GT)
    temp_skl_fp = (1-predict_img_) * skel_GT
    temp_skl_fn = predict_img_ * (1-skel_GT)
    temp_skl_tn = predict_img_ * skel_GT
    count_skl_tp = sum(sum(temp_skl_tp))
    count_skl_fp = sum(sum(temp_skl_fp))
    count_skl_fn = sum(sum(temp_skl_fn))
    count_skl_tn = sum(sum(temp_skl_tn))
    temp_pseudo_p = count_skl_tp / (count_skl_fp + count_skl_tp + 1e-4) 
    temp_pseudo_r = count_skl_tp / (count_skl_fn + count_skl_tp + 1e-4) 
    temp_pseudo_f = 2 * (temp_p * temp_pseudo_r) / (temp_p + temp_pseudo_r + 1e-4)
    return temp_pseudo_f


def DRD(predict_img_,GT_img_):
    xm = GT_img_.shape[0]
    ym = GT_img_.shape[1]
    blkSize=8 
    MaskSize=5 
    u0_GT1 = np.zeros((xm+2,ym+2)) 
    u0_GT1[1 : xm + 1, 1 : ym + 1] = GT_img_
    intim = np.cumsum(np.cumsum(u0_GT1, 0), 1)
    NUBN = 0
    blkSizeSQR = blkSize * blkSize
    counter = 0
    for i in range(1,(xm - blkSize + 1),blkSize): 
        for j in range(1,(ym - blkSize + 1),blkSize): 
            
            blkSum=intim[i + blkSize - 1, j + blkSize - 1] - intim[i - 1, j + blkSize - 1] - intim[i + blkSize - 1, j - 1] + intim[i - 1, j -1] 
            if blkSum == 0:
                pass
            elif blkSum == blkSizeSQR: 
                counter += 1;
                pass
            else: 
                NUBN = NUBN + 1
    wm = np.zeros((MaskSize, MaskSize))
    ic = int((MaskSize + 1) / 2 ) 
    jc = ic 
    for i in range(0,MaskSize): 
        for j in range(0,MaskSize): 
            num = math.sqrt((i+1 - ic) * (i+1 - ic) + (j+1 - jc) * (j+1 - jc))
            if num == 0: 
                wm[i, j]=0
            else: 
                wm[i, j] = 1 / num
    wnm = wm / sum(sum(wm)) 
    u0_GT_Resized = np.zeros((xm + ic + 1, ym + jc + 1)) 
    u0_GT_Resized[ic-1 : xm + ic - 1, jc-1 : ym + jc - 1]= GT_img_
    u_Resized = np.zeros((xm + ic + 1, ym + jc + 1)) 
    u_Resized[ic-1 : xm + ic - 1, jc-1 : ym + jc - 1] = predict_img_
    temp_fp_Resized = (1-u_Resized) * u0_GT_Resized 
    temp_fn_Resized = u_Resized * (1-u0_GT_Resized) 
    Diff = temp_fp_Resized+temp_fn_Resized 
    Diff[Diff==0] = 0 
    Diff[Diff>0] = 1 
    xm2 = Diff.shape[0] 
    ym2 = Diff.shape[1] 
    SumDRDk = 0
    def my_xor_infile(u_infile, u0_GT_infile): 
        temp_fp_infile = (1-u_infile) * u0_GT_infile 
        temp_fn_infile = u_infile * (1-u0_GT_infile) 
        temp_xor_infile = temp_fp_infile + temp_fn_infile 
        temp_xor_infile[temp_xor_infile==0] = 0 
        temp_xor_infile[temp_xor_infile>0] = 1 
        return temp_xor_infile
    for i in range(ic-1,xm2 - ic + 1): 
        for j in range(jc-1,ym2 - jc + 1): 
            if Diff[i,j] == 1: 
                Local_Diff = my_xor_infile(u0_GT_Resized[i - ic +1 : i + ic  , j - ic+1 : j + ic ], u_Resized[i, j]) 
                DRDk = sum(sum(Local_Diff * wnm)) 
                SumDRDk = SumDRDk + DRDk       
    temp_DRD = SumDRDk / (NUBN + 1e-4)
    return temp_DRD



"""
Data Utils for Sauvola Document Binarization
"""

import os
import cv2
import numpy as np
from glob import glob
from absl import logging

def collect_binarization_by_dataset(dataset_root) :
    """Load all training samples from dataset root
    and return a dict {'dataset_name' -> [original, GT]}
    """
    all_files = glob(f'{dataset_root}/*.*')
    dataset_lut = {}
    for f in all_files :
        if 'source' in f:
            source_file = f
            target_file = f.replace('source', 'target')
            if target_file in all_files :
                logging.info(f"Found pair\n\tsource={source_file}\n\ttarget={target_file}")
            else :
                logging.warning(f"Fail to find pair\n\tsource={source_file}\n\ttarget={target_file}")
                continue
            dname = os.path.basename(f).split('_')[0]
            if dname not in dataset_lut :
                dataset_lut[dname] = []
            dataset_lut[dname].append((source_file, target_file))
    return dataset_lut

class DataGenerator :
    """Simple Data Generator that consumes paired (img, gt)
    and outputs batch of (X, Y), where
        X is of shape `output_shape + (1,)`
        Y is of shape `output_shape + (1,)`
        
    When mode='training', image flipping is applied
    
    #TODO:
        1. more data augmentations, e.g. color, size, noise, etc.
        2. balanced sampling w.r.t. dataset names
    """
    def __init__(self, data_pairs, 
                 output_shape=(256,256),
                 batch_size=64,
                 nb_batches_per_epoch=1000,
                 mode='training',
                 seed=123455,
                 minimum_text_rate=0,
                ) :
        self.data_pairs = self._read_data_if_necessary(data_pairs)
        self.nb_samples = len(data_pairs)
        self.output_shape = output_shape
        self.mode = mode
        self.minimum_text_rate = minimum_text_rate
        if mode != 'testing' :
            self.batch_size = batch_size
            self.nb_batches_per_epoch = nb_batches_per_epoch
        else :
            self.batch_size = 1
            self.nb_batches_per_epoch = self.nb_samples
        self.batch_idx = 0
        self.prng = self.get_prng(seed)
    def _read_data_if_necessary(self, data_pairs) :
        rets = []
        for src, gt in data_pairs :
            if isinstance(src, str) :
                src = cv2.imread(src, 0)
            if isinstance(gt, str) :
                gt = cv2.imread(gt, 0)
            rets.append([src, gt])
        return rets
    def get_prng(self, seed=None) :
        if (seed is not None) :
            return np.random.RandomState(seed)
        else :
            return np.random.RandomState(None)
    def __len__(self) :
        return self.nb_batches_per_epoch
    def __iter__(self) :
        return self
    def __next__(self) :
        bidx = self.batch_idx
        if (self.batch_idx >= self.nb_batches_per_epoch) :
            bidx = self.batch_idx = 0
        else :
            self.batch_idx += 1
        return self[bidx]
    def crop_sample(self, img, gt, prng, niter=1) :
        h, w = img.shape[:2]
        if self.output_shape is None :
            assert self.batch_size == 1, "ERROR: original output size is only compatible with batch_size = 1"
            return img, gt
        else :
            th, tw = self.output_shape
            if (h<th) :
                return self.crop_sample( np.row_stack([img, img]), 
                                         np.row_stack([gt, gt]),
                                         prng )
            elif (w<tw) :
                return self.crop_sample( np.column_stack([img, img]), 
                                         np.column_stack([gt, gt]),
                                         prng )
            else :
                y0 = prng.randint(0, h-th+1)
                x0 = prng.randint(0, w-tw+1)
                cim, cgt = img[y0:y0+th, x0:x0+tw], gt[y0:y0+th, x0:x0+tw]
                perc_text = np.mean(cgt < 127)
                if perc_text < self.minimum_text_rate :
                    if niter < 5 :
                        return self.crop_sample(img, gt, prng, niter+1)
                return cim, cgt
    def __getitem__(self, batch_idx) :
        if self.mode != 'testing' :
            if self.mode == 'training' :
                prng = self.prng
            else :
                prng = self.get_prng(batch_idx)
            indices = prng.randint(0, self.nb_samples, size=(self.batch_size,))
        else :
            indices = [batch_idx]
            prng = self.prng
        X, Y = [], []
        for i in indices :
            img, gt = self.data_pairs[i]
            x, y = self.crop_sample(img, gt, prng)
            if (self.mode == 'training') :
                if prng.randn() > 0 :
                    x = x[::-1]
                    y = y[::-1]
                if prng.randn() > 0 :
                    x = x[:,::-1]
                    y = y[:,::-1]
                if prng.randn() > 0 :
                    x = x.T
                    y = y.T
            X.append(x)
            Y.append(y)
        return self.postprocess_image(X), self.postprocess_label(Y)
    def postprocess_image(self, X) :
        X = [ (x-x.min())/(x.max()-x.min()+.1) for x in X]
        return np.expand_dims(np.stack(X, axis=0), axis=-1).astype('float32')
    def postprocess_label(self, Y) :
        Y = np.expand_dims(np.stack(Y, axis=0), axis=-1).astype('float32')-127
        Y = np.sign(Y)
        return Y
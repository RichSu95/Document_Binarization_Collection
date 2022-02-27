"""
Test Utils for Sauvola Document Binarization
"""

import os,yaml
if not yaml.load(open('Config.yaml', 'rb'), Loader=yaml.Loader)['Global']['use_gpu']:
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from modelUtils import create_multiscale_sauvola
from layerUtils import SauvolaLayerObjects
from dataUtils import DataGenerator, collect_binarization_by_dataset
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping,ReduceLROnPlateau
from absl import logging
from parse import parse
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--conf", default="Config.yaml",help="configuration file path")
        self.add_argument("-a", "--args", nargs='+', help="configuration arguments. e.g.: -a Train.loss=mse")
    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.conf is not None, \
            "Please specify --config=configure_file_path."
        config ={}
        if not args.args:
            pass
        else:
            for s in args.args:
                s = s.strip()
                k, v = s.split('=')
                config[k] = yaml.load(v, Loader=yaml.Loader)
        args.args=config
        return args
class AttrDict(dict):
    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))
global_config = AttrDict()
default_config = {'Global': {}}
def load_config(file_path):
    merge_options(default_config)
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files"
    merge_options(yaml.load(open(file_path, 'rb'), Loader=yaml.Loader))
    return global_config
def merge_options(config):
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in global_config
            ), "sub_keys can only be one of global_config: {}, but got: {}".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
def print_conf(conf, d=4, d_iter=5):
    for k, v in conf.items():
        if isinstance(v, dict):
            print("{}{} : ".format(d * " ", str(k)))
            print_conf(v, d + d_iter)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            print("{}{} : ".format(d * " ", str(k)))
            for value in v:
                print_conf(value, d + d_iter)
        else:
            print("{}{} : {}".format(d * " ", k, v))

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

def set_conf():
    conf = ArgsParser().parse_args()
    config = load_config(conf.conf)
    merge_options(conf.args)
    print_conf(config)
    return config

def load_pretrained_model(config, metrics=['TextAcc', 'Acc', 'F1', 'PSNR']) :
    """Load pretrained model
    """
    model_filepath=config['Global']['pretrained_model']
    for m in metrics :
        assert m in SauvolaLayerObjects.keys(), \
            f"ERROR: unsupported metric {m}, be sure to register it in `SauvolaLayerObjects`"  
    model = tf.keras.models.load_model(model_filepath,
                                  custom_objects=SauvolaLayerObjects,
                                  compile=False,
                                 )
    model.compile(config['Train']['optimizer'], loss=config['Train']['loss'], metrics=[SauvolaLayerObjects[m] for m in metrics])
    return model

def prepare_training(config=None,model_root='expt') :
    model_dir = os.path.join(model_root, config['Global']['model_name'])
    os.system('mkdir -p {}'.format(model_dir))
    logging.info(f"use expt_dir={model_dir}")
    callbacks_list=config['Train']['Callbacks']['callbacks']
    callbacks=[]
    if 'ModelCheckpoint' in callbacks_list:
        ckpt = ModelCheckpoint(filepath='{}/{}'.format(model_dir, config['Global']['model_name']) + '_E{epoch:02d}-Acc{Acc:.4f}-Tacc{TextAcc:.4f}-F{F1:.4f}-PSNR{PSNR:.2f}.h5',verbose=1, save_best_only=True, save_weights_only=False, monitor = 'F1', mode='max')
        callbacks.append(ckpt)
    if 'TensorBoard' in callbacks_list:
        tb = TensorBoard(log_dir=model_dir)
        callbacks.append(tb)
    if 'EarlyStopping' in callbacks_list:
        es = EarlyStopping(patience=config['Train']['Callbacks']['patience'])
        callbacks.append(es)
    if 'WandbCallback' in callbacks_list:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project='Binarization', name=config['Global']['model_name'],mode='disabled')
        wandb.config.update(config) 
        wb=WandbCallback()
        callbacks.append(wb)
    if 'ReduceLROnPlateau' in callbacks_list:
        lr = ReduceLROnPlateau(factor=.5, min_lr=1e-6, patience=config['Train']['Callbacks']['patience'])
        callbacks.append(lr)
    return callbacks, model_dir

def train_model():
    config=set_conf()
    callbacks,model_dir=prepare_training(config=config)
    if config['Global']['pretrained_model'] is None:
        config['Global']['pretrained_model']=''
    if not os.path.isfile(config['Global']['pretrained_model']):
        model=create_multiscale_sauvola(config=config)
        metrics=['TextAcc', 'Acc', 'F1', 'PSNR']
        for m in metrics :
            assert m in SauvolaLayerObjects.keys(), \
                f"ERROR: unsupported metric {m}, be sure to register it in `SauvolaLayerObjects`"  
        model.compile(config['Train']['optimizer'], loss=config['Train']['loss'], metrics=[SauvolaLayerObjects[m] for m in metrics])

    else:
        model=load_pretrained_model(config=config)
    train_ds = collect_binarization_by_dataset(config['Train']['dataset'])
    dg = DataGenerator(train_ds["TRAIN"], batch_size=config['Train']['batch_size'], output_shape=None, nb_batches_per_epoch=len(train_ds["TRAIN"]) // config['Train']['batch_size'], mode='training')
    hist = model.fit(x=dg,callbacks=callbacks,steps_per_epoch=len(train_ds["TRAIN"]) // config['Train']['batch_size'],epochs=config['Train']['epochs'])
    for key in hist.history:
      print(key)

if __name__=='__main__':
    train_model()
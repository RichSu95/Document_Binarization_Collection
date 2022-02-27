# SauvolaNet: Learning Adaptive Sauvola Network
For the Original repo visi the following: https://github.com/Leedeng/SauvolaNet.git


# Training

Configure `Config.yaml` file and simply run the following to start training: (Check the following for details on [Config parameters](docs/Training_Config.md))

```bash
$ python train.py
```
The following list can be viewed through --help
```bash
$ python train.py -h

usage: train.py [-h] [-c CONF] [-a ARGS [ARGS ...]]

optional arguments:
  -h, --help            show this help message and exit
  -c CONF, --conf CONF  configuration file path
  -a ARGS [ARGS ...], --args ARGS [ARGS ...]
                        configuration arguments. e.g.: -a Train.loss=mse
```

## Dataset

For **each** image there should be an image for the original image e.g. `TRAIN_image1_source.jpg`, and an image for the ground truth image e.g. `TRAIN_image1_target.jpg`

The *pattern* to match source and groundtruth images is the name before the `'_source.*'` or `'_target.*'`
Besides, all the names should begin with `'TRAIN_'`
⟶ **In Summury**, for each image there should be the following:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(`'TRAIN_<uniqueID>_source.<Extention>'`, `'TRAIN_<uniqueID>_target.<Extention>'`)


# Dependency

LineCounter is written in TensorFlow.
  
  - TensorFlow-GPU: 1.15.0
  - keras-gpu 2.2.4 
 Not tested with other versions


Alternatively, one may find the inference code bz the author using this [google colab link](https://colab.research.google.com/drive/1aGYXVRuTf1dhoKSsOCPcB4vKULtplFSA?usp=sharing).

## Citation
```
@INPROCEEDINGS{9506664,  
  author={Li, Deng and Wu, Yue and Zhou, Yicong},  
  booktitle={The 16th International Conference on Document Analysis and Recognition (ICDAR)},   
  title={SauvolaNet: Learning Adaptive Sauvola Network for Degraded Document Binarization},   
  year={2021},  
  volume={},  
  number={},  
  pages={538–553},  
  doi={https://doi.org/10.1007/978-3-030-86337-1_36}}
```

The custom training code was provided by @mohamadmansourX. For more information, see [link](https://github.com/mohamadmansourX/SauvolaNet-Training).
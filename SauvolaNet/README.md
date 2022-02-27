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


Alternatively, one may find the inference code using this [google colab link](https://colab.research.google.com/drive/1aGYXVRuTf1dhoKSsOCPcB4vKULtplFSA?usp=sharing).


| DIBCO 2009   | http://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/   |
| DIBCO 2010   | http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/ |
| DIBCO 2011   | http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/        |
| DIBCO 2012   | http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/       |
| DIBCO 2013   | http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/        |
| DIBCO 2014   | http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/  |
| DIBCO 2016   | http://vc.ee.duth.gr/h-dibco2016/benchmark/                 |
| DIBCO 2017   | https://vc.ee.duth.gr/dibco2017/                 |
| DIBCO 2018   | https://vc.ee.duth.gr/h-dibco2018/                 |
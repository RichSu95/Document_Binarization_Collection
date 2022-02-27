# robin

**robin** uses a number of open source projects to work properly:
- [Keras](https://keras.io/) - high-level neural networks API;
- [Tensorflow](https://www.tensorflow.org/) - open-source machine-learning framework;
- [OpenCV](https://opencv.org/) - a library of programming functions mainly aimed at real-time computer vision;
- [Augmentor](https://augmentor.readthedocs.io/en/master/) - a collection of augmentation algorithms;

## Installation

**robin** requires [Python](https://www.python.org/) v3.5+ to run.

Get **robin**, install the dependencies from requirements.txt, download datasets and weights.

```sh
$ git clone https://github.com/masyagin1998/robin.git
$ cd robin
$ pip install -r requirements.txt
```
## HowTo

#### Robin

**robin** consists of two main files: `src/unet/train.py`, which generates weights for U-net model from input 128x128 pairs of original and ground-truth images, and `src/unet/binarize.py` for binarization group of input document images.


##Citation
```
@misc{robin,
  title = {Robust Document Image Binarization},
  author = {Mikhail Masyagin},
  note = {Available at: \url{https://github.com/masyagin1998/robin}}
}
``` 
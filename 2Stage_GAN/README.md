# Two-Stage Generative Adversarial Networks for Document Image Binarization with Color Noise and Background Removal


A Pytorch implementation of Two-Stage Generative Adversarial Networks for Document Image Binarization described in the paper:
* [Two-stage Generative Adversarial Networks for Document Image Binarization](https://arxiv.org/abs/2010.10103)

## Environment
Install requirements.txt

## Data
Place degraded images and ground truth images in folders ./images and ./GT respectively. The original image and ground truth images should have the same file names. (Eg: ./images/A.png & ./GT/A.png)

## Model training

- Create patches
 ```bash
 (In the case of dibco)
 python make_ground_truth_dibco.py
 python make_ground_truth_512_dibco.py
 ```


- Train a model per datasets
```bash
1) python train_step1_all.py 
2) python predict_for_step2.py
3) python train_step2.py
4) python train_step2_resize.py
```

## Model evaluation
Place degraded images in ./image_test and masks in ./mask_test
```bash
python eval_step2_all.py
```

- You can download trained weights from the original authors via Dropbox. The link of the files as follows.
- [Link](https://www.dropbox.com/sh/vm9mvtsaek9620s/AAAtztL7a_Z-h6J4spd-Cpbua?dl=0)


##Citation
```
@article{DBLP:journals/corr/abs-2010-10103,
  author    = {Sungho Suh and Jihun Kim and mPaul Lukowicz and Yong Oh Lee},
  title     = {Two-Stage Generative Adversarial Networks for Document Image Binarization
               with Color Noise and Background Removal},
  journal   = {CoRR},
  volume    = {abs/2010.10103},
  year      = {2020},
  url       = {https://arxiv.org/abs/2010.10103},
  eprinttype = {arXiv},
  eprint    = {2010.10103},
  timestamp = {Mon, 26 Oct 2020 15:39:44 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2010-10103.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
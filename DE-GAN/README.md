# DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement
## Description
This is an implementation for the paper [DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement](https://ieeexplore.ieee.org/document/9187695)
DE-GAN is a conditional generative adversarial network designed to enhance the document quality before the recognition process. It could be used for document cleaning, binarization, deblurring and watermark removal. The weights are available to test the enhancement. 

## Environment
install from requirements.txt file

## Training 
To train with your own data, place your degraded images in the folder "images/A/" and the corresponding ground-truth in the folder "images/B/". It is necessary that each degraded image and its corresponding gt are having the same name (could have different extentions), also, the number images  should be the same in both folders.
```bash
python train.py 
```

## Evaluation
Download the trained weghts provided by the authors to directly use the model for document enhancement, save these weights in the subfolder named weights, in the DE-GAN folder. The link to download the weights is : https://drive.google.com/file/d/1J_t-TzR2rxp94SzfPoeuJniSFLfY3HM-/view?usp=sharing

### Document binarization
```bash
python enhance.py binarize ./image_to_binarize ./directory_to_binarized_image
```

### Document deblurring
```bash
python enhance.py deblur ./image_to_deblur ./directory_to_deblurred_image
```


### Watermark removal
```bash
python enhance.py unwatermark ./image_to_unwatermark ./directory_to_unwatermarked_image
```

## Citation
```
@ARTICLE{Souibgui2020,
  author={Mohamed Ali Souibgui  and Yousri Kessentini},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement}, 
  year={2020},
  doi={10.1109/TPAMI.2020.3022406}}
```
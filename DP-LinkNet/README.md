# DP-LinkNet: A convolutional network for historical document image binarization


The implementation and the pre-trained models by the authors  are available at https://github.com/beargolden/DP-LinkNet.

### Environment
Install requirements.txt file

### Structure
Place degraded images in "./dataset/" and ground truth in "./dataset_GT/" folders. Make sure the degraded image and the corresponding ground truths have the same file names. (Ex: "./dataset/A.png" and "./dataset_GT/A.png")

### Training
First, to create patches, run
```bash
python data_prepare.py
```
A new folder names "./data/" will contain the patched pair of images. Then run
```bash
python train.py
```

### Testing
The trained models will be stored in the "./weights/" folder.
Enter the folder containing test images into the test.py file.
```bash
python test.py
```
Alternative:
Download pre-trained LinkNet34, D-LinkNet34, and DP-LinkNet34 models provided by the authors
[Google Drive] Link: https://drive.google.com/file/d/1A3QeiPwjQM2wUwMwyyWSgT9mzsEx4Q-T/view?usp=sharing

##Citation
```
@article{tiis:24643, 
title="DP-LinkNet: A convolutional network for historical document image binarization", 
author="Wei Xiong and Xiuhong Jia and Dichun Yang and Meihui Ai and Lirong Li and Song Wang and ", 
journal="KSII Transactions on Internet and Information Systems", 
DOI={10.3837/tiis.2021.05.011}, 
volume={15}, 
number={5}, 
year="2021", 
month={May}, 
pages={1778-1797}
}
```
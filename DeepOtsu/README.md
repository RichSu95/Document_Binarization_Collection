# DeepOtsu
The implementation and the pre-trained models by the authors  are available at http://www.ai.rug.nl/~sheng/DeepOtsu_tf.tar.gz.

### Environment
Install requirements.txt file

### Structure
-dataset: the directory of the input images
-imgtype: image type, such as png,jpg,tiff, etc.
-overlap: the overlap of crop small patches (default: 0.1) The larger of this value (maximum: 1.0), the smaller of the number of patches and the faster for running.
-multiscale: whether use multiscale patches (default: True) Set it to False will fast the running.
-num_block: (number of UNets used in the program, default: 6), but 3 might also good. The smaller of this number, the faster of running.

### Testing
```bash
python binTestImprovedFusion.py --dataset directory_or_image/ --imgtype png --overlap 0.1 --multiscale True --num_block 6
```

## Citation
```
@article{deepotsu,
   title={DeepOtsu: Document enhancement and binarization using iterative deep learning},
   volume={91},
   ISSN={0031-3203},
   url={http://dx.doi.org/10.1016/j.patcog.2019.01.025},
   DOI={10.1016/j.patcog.2019.01.025},
   journal={Pattern Recognition},
   publisher={Elsevier BV},
   author={He, Sheng and Schomaker, Lambert},
   year={2019},
   month={Jul},
   pages={379–390}
}
```
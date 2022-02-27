# Config Description

The following list can be accessed by the terminal

|         FLAG             |     Supported script    |        Use        |      Defaults       |         Note         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          `-c`             |      ALL       |  Specify configuration file to use  |  `Config.yaml`  |  The Config file used in initializing some parameters in the pipeline |
|          `-a`              |      ALL       |  override configuration arguments in the config file  |  `None`  |  Using -a has higher priority than the configuration file selected with -c. E.g: `-a Global.model_name= Sauvola2` |

## INTRODUCTION TO PARAMETERS OF CONFIGURATION FILE

### Global

|         Parameter             |            Use                |      Defaults       |            Note            |
| :----------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      use_gpu             |    Set using GPU or not           |       `True`        |                Used to indicate whether to use GPU or not                 |
|      model_name   |    Specify the name of the run to be using in naming the model           |       `Sauvalo_Finetune`          |                This will be used in naming the created saved model name besides to Wandb Initializations if Used                 |
|      pretrained_model    |    Set the path to pretrained model         |       `pretrained_models/Sauvola_demo.h5 `         |                If path is `None` or doesn't exist, the model will **start from scratch**.  **If the path exists**, **parameters related to Architecture will be ommited** and will be **initialized from the saved model**               |



### Train

|         Parameter             |            Use            |      Defaults        |            Note             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      Optimizer        |         Set Optimizer class name          |  `adam`  |  Check [keras Optimizers](https://keras.io/api/optimizers/) for more  |
|      Loss        |         Set Loss function name          |  `hinge`  |  Check [keras Losses](https://keras.io/api/losses/) for more  |
|      Epoch        |         Set Epochs          |  `100`  |    |
|      batch_size        |         Set Batch Size          |  `1`  |    |
|      dataset        |         Set the Dataset path          |  `Dataset`  |  Datset Folder should contain all images with names=`TRAIN_*`, and for each image there should be ground truth and source having same name but one ending with `_source.png` and groundtruth with `_target.png` e.g. for one image: `Bickely2010_H01_source.png, Bickely2010_H01_target.png`|
|      **Callbacks**        |         Callbacks Class          |    |    |
|      callbacks        |         Set callbacks to be used          |  `['ModelCheckpoint','TensorBoard','EarlyStopping','ReduceLROnPlateau']`  |    |
|      patience        |         Set patience to be used  in `['EarlyStopping','ReduceLROnPlateau']`        |  `15`  |  Note in ReduceLROnPlateau the patiance is divied by 2  |


### Architecture
In Sauvolanet, the network is divided into four stages: SauvolaMultiWindow, Pixelwise Window Attention (PWA), and Adaptive Sauolva Threshold (AST)

|         Parameter             |            Use            |      Defaults        |            Note             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      **SauvolaMultiWindow**        |         SauvolaMultiWindow Class          |    |    |
|      window_size_list        |         Sets the windows list sizes          |  `[3,5,7,11,15,19]`  |  `[int]`, the used window sizes to compute Sauvola based thresholds  |
|      norm_type        |         SauvolaMultiWindow Class          |  `'bnorm'`  |  `str`, one of `{'inorm', 'bnorm'}`, the normalization layer used in the conv_blocks {`inorm`: InstanceNormalization, `bnorm`: BatchNormalization}  |
|      activation        |         Set the activation class name           |  `'relu'`  |  `str`, the used activation function inside the SauvolaMultiWindow Convolutions |
|      base_filters        |         Sets the number of base filters          |  `4`  |  the number of base filters used in conv_blocks, i.e. the 1st conv uses `base_filter` of filters the 2nd conv uses `2*base_filter` of filters and Kth conv uses `K*base_filter` of filters  |
|      init_k        |         Set param k in Sauvola binarization          |  `0.2`  |  Initialize param k in Sauvola binarization  |
|      init_R        |         Set param R in Sauvola binarization          |  `0.5`  |  Initialize param R in Sauvola binarization  |
|      train_k        |         Set param k training flag          |  `True`  |  whether or not train the param k in Sauvola binarization  |
|      train_R        |         Set param R training flag          |  `True`  |  whether or not train the param R in Sauvola binarization  |
|      **DifferenceThresh**        |         DifferenceThresh Class          |    |  |
|      init_alpha        |         Set param alpha in Sauvola binarization          |  `16`  |  Initialize param alpha in Sauvola binarization  |
|      train_alpha        |         Set param alpha training flag          |  `True`  |  whether or not train the param alpha in Sauvola binarization  |

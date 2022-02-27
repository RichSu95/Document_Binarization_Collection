"""
Model Utils for Sauvola Document Binarization
"""
import os
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from layerUtils import *
from tensorflow.keras.models import *
from absl import logging

def conv_block(x, 
               filters, 
               kernel_size=(3,3), 
               dilation_rate=(1,1), 
               strides=(1,1), 
               norm_type='bnorm', 
               suffix='',
               activation='relu',
               **conv_kwargs,
              ) :
    """create a simple conv2d block of structure
        x -> [Conv2D -> Norm -> Activation] -> y
    
    # INPUTS:
        x: tf.tensor, 
            the conv_block's input tensor
        filters: int,
            the number of filters used in conv2D
        kernel_size: tuple of int, 
            the kernel size used in conv2D
        dilation_rate: tuple of int
            the dilation rate used in conv2D
        strides: tuple of int
            the strides used in conv2D
        norm_type: str, {'inorm' or 'bnorm'}
            either use the classic batchnormalization
            or use the instance normalization
        suffix: str,
            the suffix used in block layer naming
            to avoid duplicated layer names
        activation: str,
            the used activation function
        **conv_kwargs: 
            additional kwargs pass to conv2D
    # OUTPUTS:
        y: tf.tensor,
            the conv_block's output tensor
    """
    y = Conv2D(filters, 
               kernel_size, 
               padding='same', 
               dilation_rate=dilation_rate, 
               strides=strides, 
               name=f'conv{suffix}',
               **conv_kwargs,
              )(x)
    if norm_type == 'bnorm' :
        y = BatchNormalization(axis=-1, 
                               name=f'bnorm{suffix}')(y)
    elif norm_type == 'inorm' :
        y = InstanceNormalization(name=f'inorm{suffix}')(y)
    else :
        raise NotImplementedError(f"ERROR: unknown normalization type {norm_type}")
    y = Activation(activation, 
                   name=f'relu{suffix}')(y)
    return y

def create_multiscale_sauvola(window_size_list=[3,5,7,11,15,19], 
                              train_k=False, 
                              train_R=False, 
                              train_alpha=False, 
                              norm_type='bnorm',
                              base_filters=4,
                              img_range=(0.,1.),
                              config=None) :
    """Create a multiscale Sauvola binarization model
    
    # INPUTS:
        window_size_list: list of int,
            the used window sizes to compute Sauvola based thresholds
        train_k: bool,
            whether or not train the param k in Sauvola binarization
        train_R: bool,
            whether or not train the param R in Sauvola binarization
        train_alpha: bool,
            whether or not train the alpha param to scale outputs
        norm_type: str, one of {'inorm', 'bnorm'}
            the normalization layer used in the conv_blocks
            `inorm`: InstanceNormalization
            `bnorm`: BatchNormalization
        base_filters: int,
            the number of base filters used in conv_blocks
            i.e. the 1st conv uses `base_filter` of filters
            the 2nd conv uses `2*base_filter` of filters
            and Kth conv uses `K*base_filter` of filters
        img_range: tuple of floats
            the min and max values of input image tensor
        config: dict
            contains all the above and additional values #TODO include them here
    
    # NOTE:
    Please refer the below link for the meaning of k and R in Sauvola Binarization
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_sauvola
    
    This MultiScale Sauvola is of the network architecture as follows
    
    img01_inp -> [SauvolaMultiWindow] -> thresh5D ------|
              -> [WindownAttentionBranch] -> winAtt5D --|-> [attention] -> th -> diff

    """
    SMWConf=config['Architecture']['SauvolaMultiWindow']
    im_inp = Input(shape=(None,None,1), name='img01_inp')
    th = SauvolaMultiWindow(window_size_list=window_size_list, 
                            train_k=train_k, 
                            train_R=train_R, 
                            name='sauvola',
                            SMWConf=SMWConf,)(im_inp)
    # attention branch
    n = len(SMWConf['window_size_list'])
    filters = SMWConf['base_filters']
    t = int(np.ceil(np.log2(max(SMWConf['window_size_list'])))) - 1
    # 1st block
    f = conv_block(im_inp, 
                   filters, 
                   activation=SMWConf['activation'],
                   suffix=0, 
                   norm_type=SMWConf['norm_type'])
    # later blocks
    for k in range(t) :
        filters += base_filters
        f = conv_block(f, 
                       filters, 
                       activation=SMWConf['activation'],
                       dilation_rate=(2,2), 
                       suffix=f'{k+1:d}', 
                       norm_type=SMWConf['norm_type'])
    # attention
    f = Conv2D(n,
               (3,3), 
               padding='same', 
               activation='softmax', 
               name='conv_att')(f)
    f = Permute((3,1,2), 
                name='time1')(f)
    th = Lambda(lambda v: K.sum(K.expand_dims(v[0], axis=-1) * v[1], 
                                axis=1, 
                                keepdims=False), 
                name='attention')([f, th])
    diff = DifferenceThresh(img_min=img_range[0], 
                            img_max=img_range[1], 
                            init_alpha=16., 
                            train_alpha=train_alpha,DTConf=config['Architecture']['DifferenceThresh'])([im_inp, th])
    name = "_".join(['w'+'.'.join([f'{w}' for w in window_size_list]), f'k{int(train_k)}', f'R{int(train_R)}', f'a{int(train_alpha)}', f'{norm_type}'])
    sauvola = Model(inputs=im_inp, outputs=diff, name='Sauvola_v3_att_' + name)
    return sauvola

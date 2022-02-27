"""
Layer Utils for Sauvola Document Binarization
"""
import os
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.initializers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
import numpy as np 
import tensorflow as tf
from absl import logging

################################################################################
# Keras Layers
################################################################################
class SauvolaMultiWindow(Layer):
    """
    MultiWindow Sauvola Keras Layer
    
    1. Instead of doing Sauvola threshold computation for one window size,
       we do this computation for a list of window sizes. 
    2. To speed up the computation over large window sizes, 
       we implement the integral feature to compute at O(1).
    3. Sauvola parameters, namely, k and R, can be selected to be
       trainable or not. Detailed meaning of k and R, please refer
       https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_sauvola
    4. Default R value is made w.r.t. normalized image of range (0, 1)
    """
    def __init__(self, 
                 window_size_list=[7,15,31,63,127],
                 init_k=0.2,
                 init_R=0.5,
                 train_k=False,
                 train_R=False,
                 SMWConf=None,
                 **kwargs):
        if SMWConf is None:
            self.window_size_list = window_size_list
            self.init_k = init_k
            self.init_R = init_R
            self.train_k = train_k
            self.train_R = train_R
        else: 
            self.window_size_list = SMWConf['window_size_list']
            self.init_k = SMWConf['init_k']
            self.init_R = SMWConf['init_R']
            self.train_k = SMWConf['train_k']
            self.train_R = SMWConf['train_R']
        self.n_wins = len(self.window_size_list)
        super(SauvolaMultiWindow, self).__init__(**kwargs)
    def get_config(self) :
        base_config = super().get_config()
        config = {"window_size_list": self.window_size_list,
                  'init_k': self.init_k,
                  'init_R': self.init_R,
                  'train_k': self.train_k,
                  'train_R': self.train_R,
                 }
        return dict(list(base_config.items()) + list(config.items()))
    def _initialize_ii_buffer( self, x ) :
        """Compute integeral image
        """
        x_pad = K.spatial_2d_padding( x, ((self.max_wh//2+1,self.max_wh//2+1), (self.max_ww//2+1,self.max_ww//2+1)) )
        ii_x  = K.cumsum( x_pad, axis=1 )
        ii_x2 = K.cumsum( ii_x, axis=2 )
        return ii_x2
    def _get_max_size( self ) :
        """Compute the max size of all windows
        """
        mh, mw = 0, 0
        for hw in self.window_size_list :
            if ( isinstance( hw, int ) ) :
                h = w = hw
            else :
                h, w = hw[:2]
            mh = max( h, mh )
            mw = max( w, mw )
        return mh, mw
    def build(self, input_shape):
        self.num_woi = len( self.window_size_list )
        self.count_ii = None
        self.lut = dict()
        self.built = True
        self.max_wh, self.max_ww = self._get_max_size()
        self.k = self.add_weight(name='Sauvola_k',
                                shape=(1,self.num_woi,1,1,1),
                                dtype='float32',
                                initializer='ones',
                                trainable=self.train_k,
                                constraint=NonNeg(),
                                )
        self.R = self.add_weight(name='Sauvola_R',
                                shape=(1,self.num_woi,1,1,1),
                                dtype='float32',
                                initializer='ones',
                                trainable=self.train_R,
                                constraint=NonNeg(),
                                )
        w_list = [np.ones([1,self.num_woi,1,1,1], dtype='float32') * self.init_k,
                  np.ones([1,self.num_woi,1,1,1], dtype='float32') * self.init_R]
        self.set_weights(w_list) # <- important to set initial values
        return
        
    def _compute_for_one_size( self, x, x_ii, height, width ) :
        # 1. compute valid counts for this key
        top   = self.max_wh//2 - height//2
        bot   = top + height
        left  = self.max_ww//2 - width //2
        right = left + width
        Ay, Ax = (top, left) #self.max_wh, self.max_ww
        By, Bx = (top, right) # Ay, Ax + width
        Cy, Cx = (bot, right) #By + height, Bx
        Dy, Dx = (bot, left) #Cy, Ax
        ii_key = (height,width)
        top_0   = -self.max_wh//2 - height//2 - 1
        bot_0   = top_0 + height
        left_0  = -self.max_ww//2 - width//2 - 1
        right_0 = left_0 + width
        Ay0, Ax0 = (top_0, left_0) #self.max_wh, self.max_ww
        By0, Bx0 = (top_0, right_0) # Ay, Ax + width
        Cy0, Cx0 = (bot_0, right_0) #By + height, Bx
        Dy0, Dx0 = (bot_0, left_0) #Cy, Ax
        # used in testing, where each batch is a sample of different shapes
        counts = K.ones_like( x[:1,...,:1] )
        count_ii = self._initialize_ii_buffer( counts )
        # compute winsize if necessary
        counts_2d = count_ii[:,Ay:Ay0, Ax:Ax0] \
                  + count_ii[:,Cy:Cy0, Cx:Cx0] \
                  - count_ii[:,By:By0, Bx:Bx0] \
                  - count_ii[:,Dy:Dy0, Dx:Dx0]
        # 2. compute summed feature
        sum_x_2d = x_ii[:,Ay:Ay0, Ax:Ax0] \
                 + x_ii[:,Cy:Cy0, Cx:Cx0] \
                 - x_ii[:,By:By0, Bx:Bx0] \
                 - x_ii[:,Dy:Dy0, Dx:Dx0]
        # 3. compute average feature
        avg_x_2d = sum_x_2d / counts_2d
        return avg_x_2d
    def _compute_for_all_sizes(self, x) :
        x_win_avgs = []
        # 1. compute corr(x, window_mean) for different sizes
        # 1.1 compute integral image buffer
        x_ii = self._initialize_ii_buffer( x )
        for hw in self.window_size_list :
            if isinstance( hw, int ) :
                height = width = hw
            else :
                height, width = hw[:2]
            this_avg = self._compute_for_one_size( x, x_ii, height, width )
            x_win_avgs.append( this_avg )
        return K.stack(x_win_avgs, axis=1)
            
    def call(self, x):
        x = K.cast(x, tf.float64)
        x_2 = x**2
        E_x = self._compute_for_all_sizes(x)
        E_x2 = self._compute_for_all_sizes(x_2)
        dev_x = K.sqrt(K.maximum(E_x2 - E_x**2, 1e-6))
        T = E_x *(1. + K.cast(self.k, 'float64') * (dev_x/K.cast(self.R, 'float64')-1.))
        T = K.cast(T, 'float32')
        return T

    def compute_output_shape(self, input_shape):
        batch_size, n_rows, n_cols, n_chs = input_shape 
        return (batch_size, self.num_woi, n_rows, n_cols, n_chs)
    
class DifferenceThresh(Layer) :
    def __init__(self, 
                 img_min=0., 
                 img_max=1., 
                 init_alpha=16.,
                 train_alpha=False,
                 DTConf=None,
                 **kwargs) :
        self.img_min = img_min
        self.img_max = img_max
        if DTConf is None:
            self.init_alpha = init_alpha
            self.train_alpha = train_alpha
        else:
            self.init_alpha = DTConf['init_alpha']
            self.train_alpha = DTConf['train_alpha']
        super().__init__(**kwargs)
    def build(self, input_shapes) :
        img_shape, th_shape = input_shapes
        self.alpha = self.add_weight(name='alpha',
                            shape=(1,1,1,1),
                            dtype='float32',
                            initializer=constant(self.init_alpha),
                            trainable=self.train_alpha,
                            constraint=NonNeg(),
                            )
        return
    def call(self, inputs) :
        img, th = inputs 
        scaled_diff = (img - th) * self.alpha / (self.img_max - self.img_min)
        return scaled_diff
    def get_config(self) :
        base_config = super().get_config()
        config = {"img_min": self.img_min, 
                  "img_max": self.img_max,
                  "init_alpha": self.init_alpha,
                  "train_alpha": self.train_alpha
                 }
        return dict(list(base_config.items()) + list(config.items()))
        
class InstanceNormalization(Layer) :
    def call(self, t)  :
        t_mu = K.mean(t, axis=(1,2), keepdims=True)
        t_sigma = K.maximum(K.std(t, axis=(1,2), keepdims=True), 1e-5)
        t_norm = (t-t_mu)/t_sigma
        return t_norm
    def compute_output_shape(self, input_shape) :
        return input_shape
    
################################################################################
# Metrics
################################################################################
def TextAcc(y_true, y_pred) :
    """Text class accuracy
    """
    y_true_text = K.cast( y_true < 0, 'float32')
    y_pred_text = K.cast( y_pred < 0, 'float32')
    true_pos = y_true_text * y_pred_text
    return K.sum(true_pos, axis=(1,2,3)) / (K.sum(y_true_text, axis=(1,2,3)) + 1e-5)

def Acc(y_true, y_pred) :
    """Overall accuracy
    """
    y_true_text = K.cast( y_true < 0, 'float32')
    y_pred_text = K.cast( y_pred < 0, 'float32')
    return K.mean(binary_accuracy(y_true_text, y_pred_text), axis=(1,2))

def F1(y_true, y_pred) :
    """Fmeasure for the text class
    """
    y_true_text = K.cast( y_true < 0, 'float32')
    y_pred_text = K.cast( y_pred < 0, 'float32')
    tp = K.sum(y_true_text * y_pred_text, axis=(1,2,3))
    tn = K.sum((1-y_true_text) * (1-y_pred_text), axis=(1,2,3))
    fp = K.sum((1-y_true_text) * y_pred_text, axis=(1,2,3))
    fn = K.sum(y_true_text * (1-y_pred_text), axis=(1,2,3))
    precision = tp / (tp + fp + 1.)
    recall = tp / (tp + fn + 1.)
    Fscore = 2/(1./(precision + 1e-5) + 1./(recall + 1e-5))
    return Fscore

def PSNR(y_true, y_pred) :
    """Overall PSNR
    """
    y_true_text = K.cast( y_true < 0, 'float32')
    y_pred_text = K.cast( y_pred < 0, 'float32')
    psnr = -10. * K.log(K.mean(MSE(y_true_text, y_pred_text), axis=(1,2))) / K.log(10.)
    return psnr

################################################################################
# Others
################################################################################       
def prepare_training(model_name, model_root='expt', patience=15) :
    model_dir = os.path.join(model_root, model_name)
    os.system('mkdir -p {}'.format(model_dir))
    logging.info(f"use expt_dir={model_dir}")
    ckpt = ModelCheckpoint(filepath='{}/{}'.format(model_dir, model_name) + '_E{epoch:02d}-Acc{val_Acc:.4f}-Tacc{val_TextAcc:.4f}-F{val_F1:.4f}-PSNR{val_PSNR:.2f}.h5',
                           verbose=1, save_best_only=True, save_weights_only=False,)
    tb = TensorBoard(log_dir=model_dir)
    es = EarlyStopping(patience=patience)
    lr = ReduceLROnPlateau(factor=.5, min_lr=1e-6, patience=patience//2)
    return [ckpt, tb, es], model_dir

SauvolaLayerObjects = {
    'TextAcc': TextAcc,
    'Acc': Acc,
    'F1': F1,
    'PSNR': PSNR,
    'InstanceNormalization': InstanceNormalization,
    'DifferenceThresh': DifferenceThresh,
    'SauvolaMultiWindow': SauvolaMultiWindow,
}

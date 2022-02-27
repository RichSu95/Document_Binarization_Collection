"""
Test Utils for Sauvola Document Binarization
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np 
from layerUtils import SauvolaLayerObjects
from dataUtils import DataGenerator, collect_binarization_by_dataset
from absl import logging
from parse import parse

def prepare_inference(model_filepath, metrics=['TextAcc', 'Acc', 'F1', 'PSNR']) :
    """Load inference model from disk and prepare evaluation metrics
    """
    for m in metrics :
        assert m in SauvolaLayerObjects.keys(), \
            f"ERROR: unsupported metric {m}, be sure to register it in `SauvolaLayerObjects`"  
    model = tf.keras.models.load_model(model_filepath,
                                  custom_objects=SauvolaLayerObjects,
                                  compile=False,
                                 )
    model.compile('adam', loss='hinge', metrics=[SauvolaLayerObjects[m] for m in metrics])
    return model

def evaluate_on_datasets(sauvola_model, test_datasets, dataset_lut, output_dir) :
    """Evaludate the given sauvola binarization model on the required dataset
    and save results as a .csv in output_dir
    """
    all_res = []
    for this in test_datasets :
        logging.info(f"now evaluate dataset {this}")
        # 1. prepare dataset samples
        data_pairs = dataset_lut[this]
        # 2. create data generator
        eval_dataset = DataGenerator(data_pairs,
                                    output_shape=None,
                                    mode='testing')
        L = len(eval_dataset)
        logging.info(f"successfully load dataset {this} with {L} samples")
        # 3. run evaluation
        ret = sauvola_model.evaluate_generator(eval_dataset, L, verbose=1)
        # 4. collect results
        res = dict(zip(sauvola_model.metrics_names, ret))
        res['Dataset'] = this
        res['#Samples'] = L
        all_res.append(res)
    # 5. save as a data frame
    csv_headers = ['Dataset', '#Samples'] + sauvola_model.metrics_names
    df = pd.DataFrame(all_res, columns=csv_headers)
    pd.set_option('display.float_format','{:.4f}'.format)
    print(f"INFO: successfully evaluated model {sauvola_model.name}")
    print(df)
    csv_file = os.path.join(output_dir, sauvola_model.name + '.csv')
    df.to_csv(csv_file, index=False)
    print(f"INFO: successfully dump evaluation results to {csv_file}")
    return


def find_best_model(model_dir, criterion='F1', lower_is_better=False) :
    """Find the best model in a model_dir according to the given criterion
    """
    tag_lut = {'F1': '-F{F1:.4f}',
               'Acc': '-Acc{Acc:.4f}',
               'TextAcc': '-Tacc{TextAcc:.4f}',
               'PSNR': '-PSNR{PSNR:.4f}',
              }
    assert criterion in criterion, f"ERROR: unknown criterion={criterion}"
    fmt = '{model_name}_E{epochs:d}{dont_care}' + tag_lut[criterion] + '{dontcare2}'
    best_model_file = None
    model_basename = os.path.basename(model_dir)
    if lower_is_better :
        is_better = lambda x, y : x < y
        best_loss = np.inf
    else :
        is_better = lambda x, y : x > y
        best_loss = -np.inf
    # loop over all weight files and find the one with the lowest loss
    print("INFO: seek best models in", model_dir)
    for f in os.listdir(model_dir) :
        if f.endswith('.h5') :
            lut = parse(fmt, f).named
            loss = lut[criterion]
            if is_better(loss, best_loss) :
                best_loss = loss
                best_model_file = os.path.join(model_dir, f)
    print("INFO: found best weight", best_model_file)
    return best_model_file

def auto_evaluate_best_on_dataset(model_dir, dataset_root, test_datasets, criterion='F1', lower_is_better=False) :
    # 1. find the best model
    model_filepath = find_best_model(model_dir, criterion=criterion, lower_is_better=lower_is_better)
    # 2. prepare data
    dataset_lut = collect_binarization_by_dataset(dataset_root)
    if model_filepath is not None :
        # 3. prepare inference
        model = prepare_inference(model_filepath)
        # 4. evaluate datasets
        evaluate_on_datasets(model, test_datasets, dataset_lut, model_dir)
    else :
        logging.warning("No valid pretrained model is found")
    return
    
    
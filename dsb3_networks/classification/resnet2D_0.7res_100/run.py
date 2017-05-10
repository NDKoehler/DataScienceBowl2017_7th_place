import sys
import os
import logging
from datetime import datetime
import tensorflow as tf
import json
import shutil

sys.path.append('../tf_scripts/')
sys.path.append('../tf_scripts/architecture/')

from tools.basic_logging import initialize_logger
from resnet2D_model import resnet2D
from modules import metrics
from modules.train_singlegpu import train
from io_modules.list_iterator_classification import List_Iterator
from modules import image_summaries
#from hellsicht.tensorflow.modules.train_multigpu import train


from config_2Dfinal import H

def train_net():

    # Create a new train directory, where to save all.log and config.json
    #H['output_dir'] = 'output_dir/train_dir/%s' % datetime.now().strftime('%Y_%m_%d_%H.%M')
    if not tf.gfile.Exists(H['output_dir']):
        tf.gfile.MakeDirs(H['output_dir'])
    with open(H['output_dir'] + '/config.json', 'w') as conf_file:
        json.dump(H, conf_file, indent = 4)
    
    shutil.copy('../tf_scripts/architecture/model_def/resnet2D.py',H['output_dir']+'/resnet2D.py')
    shutil.copytree('../tf_scripts/architecture/',H['output_dir']+'/architecture')
    shutil.copytree('../tf_scripts/io_modules/',H['output_dir']+'/io_modules')    
    initialize_logger(folder=H['output_dir'])


    with tf.Graph().as_default(), tf.device('/cpu:0'):

        train_data_iter = List_Iterator(H, img_lst = H['train_lst'],
            img_shape = H['image_shape'],
            label_shape = H['label_shape'],
            batch_size = H['batch_size'],
            num_preprocess_threads = 4,
            shuffle = True,
            is_training = True,
            )

        valid_data_iter = List_Iterator(H, img_lst = H['val_lst'], 
            img_shape = H['image_shape'],
            label_shape = H['label_shape'],
            batch_size = H['batch_size'],
            num_preprocess_threads = 4,
            shuffle = True,
            is_training = False,
            )

        model = resnet2D

        update_scopes = [] 
        #update_scopes.append('logits')

        # Loss operations 

        loss_op = metrics.logloss


        # Additional Evaluation metrics
        metric_ops = None# [metrics.MSE_metric]

        H['train_image_summary'] = image_summaries.classification_image_summary
        H['validation_image_summary'] = image_summaries.classification_image_summary

        H['model_graph'] = model
        H['loss'] = loss_op
        H['metrics'] = metric_ops
        H['train_scopes'] = update_scopes
        H['train_iter'] = train_data_iter
        H['valid_iter'] = valid_data_iter
        H['VARIABLES_TO_RESTORE'] = tf.contrib.slim.get_variables_to_restore()
        H['UPDATE_OPS_COLLECTION'] = tf.GraphKeys.UPDATE_OPS

        args = []
        train(*args, **H)

def main(argv=None):
    train_net()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in H['gpus']])
    tf.app.run()

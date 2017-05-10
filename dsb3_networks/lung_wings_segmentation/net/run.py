import sys
import os
import logging
from datetime import datetime
import tensorflow as tf
import json

sys.path.append('./subscripts/')
from basic_logging import initialize_logger
import metrics
from train_singlegpu import train
#from t.tensorflow.modules.eval import eval
#from hellsicht.tensorflow.modules.predict import predict
from  proto_iterator_segmentation import Proto_Iterator
import image_summaries
sys.path.append('./subscripts/architectures/')
import unet_model as unet_model


from config import H



def train_net():
  
    # Create a new train directory, where to save all.log and config.json
    #H['output_dir'] = 'output_dir/train_dir/%s' % datetime.now().strftime('%Y_%m_%d_%H.%M')
    if not tf.gfile.Exists(H['output_dir']):
        tf.gfile.MakeDirs(H['output_dir'])
    with open(H['output_dir'] + '/config.json', 'w') as conf_file:
        json.dump(H, conf_file, indent = 4)
    initialize_logger(folder=H['output_dir'])


    with tf.Graph().as_default(), tf.device('/cpu:0'):

        train_data_iter = Proto_Iterator(H, record_file = H['train_lst'],
            img_shape = H['image_shape'],
            label_shape = H['label_shape'],
            batch_size = H['batch_size'],
            num_preprocess_threads = 4,
            shuffle = True,
            is_training = True,
            )

        valid_data_iter = Proto_Iterator(H, record_file = H['val_lst'], 
            img_shape = H['image_shape'],
            label_shape = H['label_shape'],
            batch_size = H['batch_size'],
            num_preprocess_threads = 4,
            shuffle = True,
            is_training = False,
            )
        
        model = unet_model.unet

        update_scopes = [] 
        #update_scopes.append('logits')

        # Loss operations 
        loss_op = metrics.jaccard_separated_channels
        # Additional Evaluation metrics
        metric_ops = metrics.jaccard_separated_channels_metric

        H['train_image_summary'] = image_summaries.segmentation_image_summary
        H['validation_image_summary'] = image_summaries.segmentation_image_summary


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
    #eval_net()
    
    # predict_net()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in H['gpus']])
    tf.app.run()

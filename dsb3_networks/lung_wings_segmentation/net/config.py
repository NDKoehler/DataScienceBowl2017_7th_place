from collections import defaultdict
from datetime import datetime
import json
import tensorflow as tf
import os, sys
import pandas as pd

#config dic
H = defaultdict(lambda: None)

#All possible config options:
H['optimizer'] = 'MomentumOptimizer'#'RMSPropOptimizer'
H['learning_rate'] = 0.001
H['momentum'] = 0.99 #0.99
H['kernel_num'] = 32#32

H['unet_type'] = 'standard' #standard double_conv, #cross_convs
H['model_name'] = 'lung_wings_segmentation'

H['dropout_keep_prob'] = 0.8

H['gpus'] = [0]
H['gpu_fraction'] = None

H['num_classes'] = 2

# class_id = int(sys.argv[1])

out_name = 'lung_wings_segmentation'

H['pretrained_checkpoint_dir'] = ''
H['output_dir'] = 'output_dir/'+out_name+'/'
H['predictions_dir'] = './output_dir/'+out_name+'/predictions/va/'

H['allow_soft_placement'] = True
H['log_device_placement'] = False
H['max_steps'] = 1020
H['MOVING_AVERAGE_DECAY'] = 0.9
H['BATCH_NORM_CENTER'] = True
H['BATCH_NORM_SCALE'] = True
H['weights_initializer'] = 'xavier_initializer_conv2d' #'xavier_initializer', 'xavier_initializer_conv2d', 'truncated_normal_initializer' 
H['summary_step'] = 10
# list iterator

H['train_lst'] = '../data/lsts+records/tr.tfr'
H['val_lst'] = '../data/lsts+records/va.tfr'
#H['val_lst'] = '../data/predictions_images/slice.lst'
# images

H['in_image_shape'] = [128,128,  1] #256
H['in_label_shape'] = [128,128,  1] #256

H['image_shape'] = [128, 128, 1] #384
H['label_shape'] = [128, 128, 1] #384
H['batch_size'] = 16#3

H['rand_flip_lr']=True
H['rand_flip_ud']=True
H['rand_rot'] =True
H['crop_min'] = 90
H['crop_max'] = 128
H['brightness'] = False#0.2
H['contrast_lower'] = False#0.75#0.5
H['contrast_upper'] = False#.25#1.5  


H['save_step'] = 30

H['tr_num_examples'] = len(pd.read_csv('../data/lsts+records/tr.lst', header=None, sep = '\t'))
H['va_num_examples'] = len(pd.read_csv('../data/lsts+records/va.lst', header=None, sep = '\t'))


# RMSPROP optimizer params
H['RMSPROP_EPSILON'] = 1.0
H['RMSPROP_DECAY'] = 0.9
H['num_epochs_per_decay'] = 30
H['learning_rate_decay_factor'] = 0.16

# Augmentation params
#H['rand_crop_train'] = True
#H['crop_min'] = 0.9
#H['crop_max'] = 1.0
#H['rand_crop_val'] = True
#H['crop_val'] = 1.0
#H['lrflip'] = True
#H['udflip'] = True
#H['rand_rot'] = True
#H['min_rot_angle'] = 10 #degree 
#H['max_rot_angle'] = 10 #degree 



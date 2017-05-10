from collections import defaultdict
from datetime import datetime
import json
import tensorflow as tf
import os, sys
import pandas as pd

#config dic
H = defaultdict(lambda: None)

#All possible config options:
H['optimizer'] = 'MomentumOptimizer'
H['learning_rate'] = 0.001
H['momentum'] = 0.99
H['kernel_num'] = 32

H['unet_type'] = 'double_conv'
H['model_name'] = 'unet'

H['dropout_keep_prob'] = 1.0

H['gpus'] = [0]
H['gpu_fraction'] = 0.9

H['num_classes'] = 2

out_name = '128x128_5Channels_mutliview_stage2/'


H['pretrained_checkpoint_dir'] = 'output_dir/128x128_5Channels_mutliview_stage1/' 
H['output_dir'] = 'output_dir/'+out_name+'/'
H['predictions_dir'] = './output_dir/'+out_name+'/predictions/va/'

H['allow_soft_placement'] = True
H['log_device_placement'] = False


H['max_steps'] = 101

H['MOVING_AVERAGE_DECAY'] = 0.9
H['BATCH_NORM_CENTER'] = True
H['BATCH_NORM_SCALE'] = True
H['weights_initializer'] = 'xavier_initializer_conv2d'
H['summary_step'] = 10
# list iterator

lst_base_path = '../../../datapipeline_final/LUNA16_0/gen_nodule_seg_data/'
H['train_lst'] = lst_base_path + '/tr_nodule_seg_data_DF.csv'
H['train_npy_path'] = lst_base_path + '/arrays/tr.npy'
H['val_lst'] = lst_base_path + '/va_nodule_seg_data_DF.csv'
H['val_npy_path'] = lst_base_path +'/arrays/va.npy'


H['load_lsts_fraction'] = False#

H['load_in_ram'] = True

H['in_image_shape'] = [128, 128, 5] 
H['in_label_shape'] = [128, 128, 1]

H['image_shape'] = [128, 128, 5]
H['label_shape'] = [128, 128, 1]

H['batch_size'] = 32

#Stuff
H['num_nodule_free_per_batch'] = 2
H['class1_weight'] = 1.0


H['staged_intensities'] = False 
H['compensation'] = False
H['min_prio'] = 3

#ROTATION
H['rand_flip_lr']=True
H['rand_flip_ud']=True
H['rand_rot'] = True
H['descreteRotAngles'] = False
H['min_rot_angle'] = -25
H['max_rot_angle'] = 25
H['nintyDegRot'] = True

#CROPPING
H['rand_crop'] = False

H['crop_min'] = False
H['crop_max'] = False


H['save_step'] = 10

H['tr_num_examples'] = 5000
H['va_num_examples'] = 1000


# RMSPROP optimizer params
H['RMSPROP_EPSILON'] = 1.0 
H['RMSPROP_DECAY'] = 0.9
H['num_epochs_per_decay'] = 30
H['learning_rate_decay_factor'] = 0.16

H['unity_center_radius'] = 2


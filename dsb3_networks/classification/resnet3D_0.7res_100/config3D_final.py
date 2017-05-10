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
H['momentum'] = 0.9 #0.99
H['kernel_num'] = 16 #32

H['dropout_keep_prob'] = 1.0

H['gpu_fraction'] = 0.9

H['num_classes'] = 2
H['model_name'] = 'resnet3D'

H['pretrained_checkpoint_dir'] = '../luna_resnet3D/output_dir/luna3D'#../luna_resnet2D/output_dir/gen8_20z_3rot_stage1_deep
H['output_dir'] = 'output_dir/3Dtest_c10_init' #cross_crop_retrain_zrot
H['predictions_dir'] = ''

H['allow_soft_placement'] = True
H['log_device_placement'] = False
H['max_steps'] = 70
H['MOVING_AVERAGE_DECAY'] = 0.9
H['BATCH_NORM_CENTER'] = True
H['BATCH_NORM_SCALE'] = True
H['weights_initializer'] = 'xavier_initializer' #'xavier_initializer', 'xavier_initializer_conv2d', 'truncated_normal_initializer' 
H['gpus'] = [0]
H['summary_step'] = 10

# list iterator
# H['train_lst'] = '../data/multiview-2/tr.lst'
# H['val_lst'] = '../data/multiview-2/va.lst'


H['train_lst'] = '../../../datapipeline_final/dsb3_0/interpolate_candidates_res07/tr_patients_100.lst'
H['val_lst'] = '../../../datapipeline_final/dsb3_0/interpolate_candidates_res07/va_patients_0.lst'

#H['train_lst'] = tr_path
#H['val_lst'] = va_path


H['candidate_mode'] = False

# crossed axes options - cross is centrally cropped -> layers are stacked in z-dim
H['num_crossed_layers'] = 1
H['crossed_axes'] = False#[0,1,2]
H['rand_drop_planes']=0
H['plane_mil'] = False
# y and x image_shape must be equal -> z has same shape!!!
# you can crop if the equal z,y and x in image shape are and smaller than in in_image_shape


# images
# in_image_shapes[1:] must be equal to len of crop_before_loading_in_RAM_ZminZmaxYminYmaxXminXmax
H['in_image_shape'] = [5, 64, 64, 64, 2] #256 
# not working #H['crop_before_loading_in_RAM_ZminZmaxYminYmaxXminXmax'] = [False,False,False,False,False,False] # Default = False or None
H['image_shape'] = [5, 64, 64, 64, 2]
H['label_shape'] = [1] #256
H['batch_size'] = 8


#iterator settings
H['load_in_ram'] = True
# due to time consuming operation and quality loss only rotation around one axis is processed randomly chosen
H['rand_rot_axes'] = [0,1,2] # 0: z, 1: y, 2: x (attention: x and y rotation lasts long)
H['rand_rot'] = True
H['degree_90_rot'] = H['rand_rot']
H['min_rot_angle'] = -45 #degree 
H['max_rot_angle'] = 45 #degree 
H['rand_mirror_axes'] = [0,1,2] # 0: z, 1: y, 2: x else False
H['rand_cropping_ZminZmaxYminYmaxXminXmax'] = [False,False,False,False,False,False] # crop within given range # default False: full range

H['save_step'] = 10 # saving checkpoint

H['tr_num_examples'] = len(pd.read_csv(H['train_lst'], header=None, sep='\t'))
H['va_num_examples'] = len(pd.read_csv(H['val_lst'], header=None, sep='\t'))

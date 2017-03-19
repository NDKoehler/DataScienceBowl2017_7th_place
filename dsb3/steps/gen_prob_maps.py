import sys
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
import tensorflow as tf
import json
import glob
import cv2
import math
import time
import json
from natsort import natsorted
import argparse
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.platform import resource_loader

from utils import ensure_dir
from master_config import H
if H['test_pipeline']:
    test_pipeline = H['test_pipeline']
else:
    test_pipeline = False

try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

np.random.seed(17) # do NOT change
#################################

prob_maps_data_type = H['gen_prob_maps_data_type'] # choose uint8 int16 or float32

# load nodule unet and checkpoint depending on python-version
nodule_seg_checkpoint_dir = H['gen_prob_maps_checkpoint_dir']
sys.path.append(nodule_seg_checkpoint_dir + '/architecture/')
from basic_logging import initialize_logger
import unet_model as nodule_unet_model

batch_sizes = H['gen_prob_maps_batch_sizes']

dataset_name = H['dataset_name']
dataset_dir = H['dataset_dir']
################################
dataset_json = json.load(open(dataset_dir + dataset_name+'_resample_lungs.json'))
dataset_json['gen_prob_maps_data_type'] = prob_maps_data_type # uint8 int16 or float32
HU_tissue_range = dataset_json['HU_tissue_range'] # uint16 or float32

def load_network(config, net, net_name, image_shape, dataset_dir, checkpoint_dir, gpu_fraction, reuse):
    out_dir = dataset_dir+'/logs/gen_prob_maps/'+str(image_shape[0])+'x'+str(image_shape[1])+'/'
    # Create a new eval directory, where to save all.log and config.json
    logging.info('Saving evaluation results to: {}'.format(out_dir))
    if not tf.gfile.Exists(out_dir):
        tf.gfile.MakeDirs(out_dir)

    with open(out_dir + '/config.json', 'w') as conf_file:
       json.dump(config, conf_file, indent = 4)
    initialize_logger(folder=out_dir)

    # Get all neccessary paramters from kwargs
    try:
        # Get model graph
        my_model_graph = net
    except:
        logging.error('(model_graph) was not provided!')
        raise KeyError('(model_graph) was not provided!')

    config['VARIABLES_TO_RESTORE'] = tf.contrib.slim.get_variables_to_restore()
    config['UPDATE_OPS_COLLECTION'] = tf.GraphKeys.UPDATE_OPS

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    init_op = tf.initialize_all_variables()

    # Define a session
    sess = tf.Session(config=tf.ConfigProto(
                      gpu_options = gpu_options,
                      allow_soft_placement=True,
                      log_device_placement=config['allow_soft_placement']))

    # Initialize all variables
    sess.run(init_op)
    data={}
    data['images'] = tf.placeholder(dtype=tf.float32, shape=[None] + image_shape, name=net_name+'_image_placeholder')

    with tf.device('/gpu:%d' % 1):
        config['dropout'] = 1.0
        logging.info('Predicting on gpu:{}'.format(1))
        # Get endpoint / or get tensor from session.graph

        out_, endpoints = my_model_graph(data=data,
                restore_logits=False,
                is_training=False,
                reuse=reuse,
                scope=net_name,
                **config)

    # Add all endpoints to predict
    config['endpoints'] = ['probs', 'logits']
    pred_ops = {}
    for key in config['endpoints']:
        if key in endpoints and key not in pred_ops:
            pred_ops[key] = endpoints[key]
        elif key in out_ and key not in pred_ops:
            pred_ops[key] = out_[key]

    if len(pred_ops) != len(config['endpoints']):
        logging.error('Not all enpoints found in the graph!: {}'.format(config['endpoints']))
        raise ValueError('Not all enpoints found in the graph!: {}'.format(config['endpoints']))

    # restore checkpoint and create saver
    if checkpoint_dir:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        var_lst = []
        var_lst = tf.all_variables()
        # print (var_lst[0].name)
        # sys.exit()
        #for var in tf.all_variables():
        #    if var.name.split('/')[0] == net_name:
        #        var_lst.append(var)
        saver = tf_saver.Saver(var_lst, write_version=tf.train.SaverDef.V2)
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        logging.info('Succesfully loaded model from %s' % (checkpoint_dir))
    else:
        raise IOError('ckpt not found: {}'.format(checkpoint_dir))
        sys.exit()
    return sess, pred_ops, data
def rotate(in_tensor, M):
    dst = cv2.warpAffine(in_tensor,M,(in_tensor.shape[1],in_tensor.shape[0]), flags=cv2.INTER_CUBIC)
    if len(dst)==2:
        dst = np.expand_dims(dst, 2)
    return dst

def rotate_3d(tensor, M, rand_axis, rand_angle):
    if rand_axis == 0:
        for i in range(tensor.shape[0]):
            tensor[i,:,:] = rotate(tensor[i,:,:], M)
    elif rand_axis == 1:
        for i in range(tensor.shape[1]):
            tensor[:,i,:] = rotate(tensor[:,i,:], M)
    elif rand_axis == 2:
        for i in range(tensor.shape[2]):
            tensor[:,:,i] = rotate(tensor[:,:,i], M)
    return tensor
def main(argv=None):

    ensure_dir(dataset_dir+'probability_maps/')

    all_patients = dataset_json['all_patients']

    if test_pipeline:
        all_patients = all_patients[:test_pipeline]

    # check if enought batch_sizes given for nodule_seg_image_shapes
    if len(H['gen_prob_maps_image_shapes']) != len(H['gen_prob_maps_batch_sizes']):
        print ('ERROR in master_config.py. Need same number of batch_sizes and image_shapes for nodule_seg')
        sys.exit()

    # sort nets in ascending size x*y
    nodule_seg_image_shapes = sorted(H['gen_prob_maps_image_shapes'], key=lambda shape: shape[0]*shape[1])

    # split patients by resampled_scan_shape for the nodule_segmentation_nets with distinct image_shapes -> save computing time
    net_numbers_with_patients = {}
    for net_num in range(len(nodule_seg_image_shapes)):
        net_numbers_with_patients[net_num] = []

    registered_patients = []
    for net_num, net_shape in enumerate(nodule_seg_image_shapes):
        if net_num==(len(nodule_seg_image_shapes)-1):
            ratio = 1.0
        else:
            ratio = H['gen_prob_maps_image_shape_max_ratio'] # for avoiding border effects net due to padding,...
        for patient in all_patients:
            if patient in registered_patients: continue
            p_shape = dataset_json['patients'][patient]['resampled_scan_shape_yxz_px']
            if (p_shape[0] < int(ratio*net_shape[0])) and (p_shape[1] < int(ratio*net_shape[1])) and (p_shape[2] < int(ratio*net_shape[0])) and (p_shape[2] < int(ratio*net_shape[1])):
                net_numbers_with_patients[net_num].append(patient)
                registered_patients.append(patient)

    if not len(registered_patients) == len(all_patients):
        print ('ERROR in registering patients for different nets')
        sys.exit()

    print ('patients distribution on nets:',[len(net_numbers_with_patients[x]) for x in list(net_numbers_with_patients.keys())])

    # define view_planes
    view_planes = sorted([x if x in ['y', 'x', 'z'] else 'ERROR' for x in natsorted(H['gen_prob_maps_view_planes'])]) #xyz loop first over z because than no img_array_swap_axes is needed
    view_planes.reverse()
    if len(view_planes) == 0:
        print ('no view_plane is determined!!!')
    elif 'ERROR' in view_planes:
        print ('wrong view_plane given: {}'.format(H['view_planes']))
        sys.exit()

    # Initialize
    # loop over  nodule_segmentation_nets with distinct image_shapes:
    for net_num, net_shape in enumerate(nodule_seg_image_shapes):
        batch_size = batch_sizes[net_num]

        # load json from checkpoint_dir
        nodule_config = json.load(open(nodule_seg_checkpoint_dir+'/config.json'))
        image_shape        = net_shape+[nodule_config['image_shape'][2]]
        nodule_label_shape = net_shape+[nodule_config['label_shape'][2]]
        prediction = np.zeros([batch_size] + list(nodule_label_shape), dtype=np.float32)
        prediction_rot = np.zeros_like(prediction)

        dataset_json['gen_prob_maps_nodule_segmentation_model_name'] = nodule_config['model_name']

        nodule_sess,     nodule_pred_ops,     nodule_data     = load_network(nodule_config, nodule_unet_model.unet, nodule_config['model_name'], image_shape, dataset_dir, nodule_seg_checkpoint_dir, H['gen_prob_maps_gpu_fraction'], True if net_num>0 else None)

        print ('predicting with net number {} - net_shape {}'.format(net_num, net_shape))
        # loop over patients in nets list
        for patient_cnt, patient in enumerate(tqdm(list(net_numbers_with_patients[net_num]))):

            patient_json = dataset_json['patients'][patient]

            # load lung crop scan [-0.25, 0.75]
            loaded_img_array = np.load(patient_json['resampled_lung_path'])
            if H['resampled_lung_data_type'] == 'int16':
                # [0, 1400] -> [-0.25,0.75] normalized and zero_centered
                loaded_img_array = ((loaded_img_array/ float((HU_tissue_range[1] - HU_tissue_range[0]))) - 0.25).astype(np.float32) # y,x,z

            prob_map = np.zeros_like(loaded_img_array, dtype=np.float32)

            # loop over all view_planes:
            for view_plane in view_planes:
                # loop_axis == 3 axis
                if view_plane == 'z':
                    org_img_array = loaded_img_array # y,x,z # loaded_img_array already has y,x,z
                elif view_plane == 'y':
                    del(img_array)
                    org_img_array = np.swapaxes(loaded_img_array.copy(),0,2) # z, x, y
                elif view_plane == 'x':
                    del(img_array)
                    org_img_array = np.rollaxis(loaded_img_array,2,0) # z, y, x # due to final view_plane iterating over no copy is needed

                # embed img in xy center in image_shape
                img_array = np.ones((image_shape[0], image_shape[1],org_img_array.shape[2]), dtype=np.float32)*(-0.25) # 'black' array
                embed_coords_y = int((img_array.shape[0]-org_img_array.shape[0])/2)
                embed_coords_x = int((img_array.shape[1]-org_img_array.shape[1])/2)
                img_array[embed_coords_y:embed_coords_y+org_img_array.shape[0], embed_coords_x:embed_coords_x+org_img_array.shape[1],:] = org_img_array

                num_batches = int(np.ceil(img_array.shape[2] / batch_size))
                batch = np.ones(([batch_size] + image_shape[:2] + [image_shape[2]]), dtype=np.float32)*(-0.25)
                batch_rot = batch.copy()
                for batch_cnt in range(num_batches):
                    batch[:]          = -0.25 # reset to black
                    batch_rot[:]      = -0.25 # reset to black
                    prediction[:]     = 0     # reset to black
                    prediction_rot[:] = 0     # reset to black

                    # fill batch
                    for cnt in range(batch_size):
                        layer_cnt = cnt+batch_size*batch_cnt
                        if layer_cnt >= img_array.shape[2]: break

                        # leave some channels above empty at top and below at bottom
                        min_z = max(0,int(layer_cnt-(image_shape[2]-1)/2))
                        max_z = min(int(layer_cnt+(image_shape[2]-1)/2)+1,img_array.shape[2])

                        batch_idx_z = [image_shape[2]-max(0,max_z-min_z), image_shape[2]-min(0,max_z-min_z)]

                        batch[cnt,:,:,batch_idx_z[0]:batch_idx_z[1]] = img_array[:,:,min_z:max_z]

                    randy = np.random.randint(0,1000)
                    # loop over view_angles
                    for view_angle in H['gen_probs_view_angles']:
                        # rot batch
                        if view_angle != 0:
                            M = cv2.getRotationMatrix2D((batch.shape[2]//2,batch.shape[1]//2), view_angle,1)
                            
                            for img_cnt in range(batch.shape[0]):
                                batch_rot[img_cnt] = rotate_3d(((batch[img_cnt].copy()+0.25)*255).astype(np.uint8), M, 2, view_angle)/255.-0.25
                        else:
                            batch_rot = batch.copy()

                        # get segmentation for noduls
                        # and reshape flat prediction to batchsize, z,x,1
                        prediction_rot = np.reshape(nodule_sess.run(nodule_pred_ops, feed_dict = {nodule_data['images']: batch_rot})['probs'], prediction.shape)
                        
                        # rotate back prediction
                        if view_angle != 0:
                            M_back = cv2.getRotationMatrix2D((prediction_rot.shape[2]//2,prediction_rot.shape[1]//2), -view_angle,1)
                            for img_cnt in range(batch.shape[0]):
                                prediction_rot[img_cnt] = np.clip(rotate_3d((prediction_rot[img_cnt]*255).astype(np.uint8), M_back, 2, -view_angle)/255.,0,1)

                        # mean over view_angles
                        prediction += prediction_rot/float(len(H['gen_probs_view_angles']))

                    # if img_shape != label_shape resizeing etc needed -> probability map has value range [0.0, 1.0]
                    if np.min(prediction)<0.0 or np.max(prediction)>1.0:
                        print ('ERROR - nodule prediciton not in value range [0.0, 1.0] for img_array {}'.format(img_array_name))
                        sys.exit()

                    # crop from embedded layers
                    prediction_embedded = prediction[:,embed_coords_y:embed_coords_y+org_img_array.shape[0],embed_coords_x:embed_coords_x+org_img_array.shape[1]]/float(len(view_planes))

                    layer_start = batch_size*batch_cnt
                    # insert prob_map layer, swaping axis in right order
                    if view_plane == 'z':
                        layer_end   = min(img_array.shape[2], layer_start+batch_size)
                        prob_map[:,:,layer_start:layer_end] += np.swapaxes(np.swapaxes(prediction_embedded[:layer_end-layer_start,:,:,0],0,1),1,2) # z,y,x -> y,x,z
                    elif view_plane == 'y':
                        layer_end   = min(img_array.shape[2], layer_start+batch_size)
                        prob_map[layer_start:layer_end,:,:] += np.swapaxes(prediction_embedded[:layer_end-layer_start,:,:,0],1,2)  # y,z,x -> y,x,z
                    elif view_plane == 'x':
                        layer_end   = min(img_array.shape[2], layer_start+batch_size)
                        prob_map[:,layer_start:layer_end,:] += np.rollaxis(prediction_embedded[:layer_end-layer_start,:,:,0],2,0) # x,z,y -> y,x,z

            if prob_maps_data_type == 'uint8':
                prob_map = (prob_map*255).astype(np.uint8)
            elif prob_maps_data_type == 'uint16':
                prob_map = (prob_map*65535).astype(np.uint16)
            patient_json['probability_map_path'] = dataset_dir+'probability_maps/'+patient+'_prob_map.npy'
            np.save(patient_json['probability_map_path'], prob_map)

        # close sess and run next net
        nodule_sess.close()

    json.dump(dataset_json, open(dataset_dir+dataset_name+'_gen_prob_maps.json', 'w'),indent=4)
    print('wrote', dataset_dir+dataset_name+'_gen_prob_maps.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pipeline', type=int, required=False, default=False, help="choose num of patients processed")
    args = parser.parse_args()
    if args.test_pipeline:
        test_pipeline = args.test_pipeline
    if H['gen_prob_maps_GPUs'] != None: os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in H['gen_prob_maps_GPUs']])
    tf.app.run()

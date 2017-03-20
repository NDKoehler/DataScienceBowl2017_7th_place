import sys
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
import json
import glob
import cv2
import math
import time
import json
from tqdm import tqdm
from .. import pipeline as pipe
from .. import tf_tools

def run(data_type,
        checkpoint_dir,
        batch_sizes,
        image_shapes,
        image_shape_max_ratio,
        view_planes,
        view_angles):
    """
    Parameters
    ----------
    data_type : {unit8, int16, float32}
        Data type of prob maps.
    """
    # check if enought batch_sizes given for image_shapes
    if len(image_shapes) != len(batch_sizes):
        raise ValueError('Need same number of batch_sizes and image_shapes for nodule_seg.')
    if len(view_planes) == 0 or len([c for c in view_planes if c not in ['x', 'y', 'z']]) > 0:
        raise ValueError('view_planes ' + str(view_planes) + 'must only contain x, y, z chars.')
    # sort nets in ascending size y * x
    image_shapes = sorted(image_shapes, key=lambda shape: shape[0]*shape[1])
    # split patients by scan_shape for the nodule_segmentation_nets with distinct image_shapes -> save computing time
    net_numbers_with_patients = [[] for n in range(len(image_shapes))]
    registered_patients = set()
    resample_lungs_result = pipe.load_step('resample_lungs')
    HU_tissue_range = resample_lungs_result['HU_tissue_range']
    for net_num, net_shape in enumerate(image_shapes):
        ratio = 1 if net_num == len(image_shapes) - 1 else image_shape_max_ratio # for avoiding border effects net due to padding, ...
        for patient in pipe.patients:
            if patient in registered_patients:
                continue
            scan_shape = resample_lungs_result[patient]['resampled_scan_shape_zyx_px']
            if (scan_shape[1] < int(ratio * net_shape[0]) and scan_shape[1] < int(ratio * net_shape[1]) 
                and scan_shape[2] < int(ratio * net_shape[0]) and scan_shape[2] < int(ratio * net_shape[1])):
                net_numbers_with_patients[net_num].append(patient)
                registered_patients.add(patient)
    if len(registered_patients) != len(pipe.patients):
        raise ValueError('Registering patients for different nets is inconsistent.')
    print('patients distribution on nets:', [len(l) for l in net_numbers_with_patients])
    view_planes = sorted(view_planes, reverse=False) # zyx loop first over z because than no img_array.swap_axes is needed
    # loop over nodule_segmentation_nets with distinct image_shapes
    patients_dict = pipe.patients_dict
    reuse_init = True
    for net_num, net_shape in tqdm(enumerate(image_shapes)):
        if len(net_numbers_with_patients[net_num]) == 0:
            continue
        reuse = None if reuse_init else True
        reuse_init = False
        batch_size = batch_sizes[net_num]
        config = json.load(open(checkpoint_dir + '/config.json'))
        image_shape = net_shape + [config['image_shape'][2]]
        label_shape = net_shape + [config['label_shape'][2]]
        prediction = np.zeros([batch_size] + list(label_shape), dtype=np.float32)
        prediction_rot = np.zeros_like(prediction)
        sess, pred_ops, data = tf_tools.load_network(checkpoint_dir, image_shape=image_shape, reuse=reuse)
        print('predicting with net {} with x-y image shape {}'.format(net_num, net_shape))
        # loop over patients in nets list
        for patient in net_numbers_with_patients[net_num]:
            org_img_array = np.load(resample_lungs_result[patient]['resampled_lung_path']) # z, y, x
            # print(org_img_array.shape)
            prob_map = np.zeros_like(org_img_array, dtype=np.float32)
            for view_plane_cnt, view_plane in enumerate(view_planes):
                if view_plane_cnt > 0: # reload
                    org_img_array = np.load(resample_lungs_result[patient]['resampled_lung_path']) # z, y, x
                if org_img_array.dtype == np.int16: # [0, 1400] -> [-0.25, 0.75] normalized and zero_centered
                    org_img_array = (org_img_array/(HU_tissue_range[1] - HU_tissue_range[0]) - 0.25).astype(np.float32)
                if view_plane == 'z':
                    org_img_array = np.rollaxis(org_img_array, 0, 2) # y, x, z
                elif view_plane == 'y':
                    org_img_array = np.swapaxes(org_img_array, 1, 2) # z, x, y
                elif view_plane == 'x':
                    org_img_array = np.swapaxes(org_img_array, 2, 0) # x, z, y ??? why do we have to do this?
                # print('org_img_array', org_img_array.shape)
                # embed img in xy center in image_shape
                img_array = (-0.25) * np.ones((image_shape[0], image_shape[1], org_img_array.shape[2]), dtype=np.float32) # 'black' array
                embed_coords_y = int((img_array.shape[0] - org_img_array.shape[0])/2)
                embed_coords_x = int((img_array.shape[1] - org_img_array.shape[1])/2)
                img_array[embed_coords_y : embed_coords_y + org_img_array.shape[0],
                          embed_coords_x : embed_coords_x + org_img_array.shape[1], :] = org_img_array
                num_batches = int(np.ceil(img_array.shape[2] / batch_size))
                batch = (-0.25) * np.ones(([batch_size] + image_shape), dtype=np.float32)
                batch_rot = batch.copy()
                for batch_cnt in range(num_batches):
                    batch[:]          = -0.25 # reset to black
                    batch_rot[:]      = -0.25 # reset to black
                    prediction[:]     = 0     # reset to black
                    prediction_rot[:] = 0     # reset to black
                    # fill batch
                    for cnt in range(batch_size):
                        layer_cnt = cnt + batch_size * batch_cnt
                        if layer_cnt >= img_array.shape[2]: break
                        # leave some channels above empty at top and below at bottom
                        min_z = max(0, int(layer_cnt - (image_shape[2]-1)/2))
                        max_z = min(int(layer_cnt + (image_shape[2]-1)/2) + 1, img_array.shape[2])
                        batch_idx_z = [image_shape[2] - max(0, max_z-min_z), image_shape[2] - min(0, max_z-min_z)]
                        batch[cnt, :, :, batch_idx_z[0]:batch_idx_z[1]] = img_array[:, :, min_z:max_z]
                    randy = np.random.randint(0,1000)
                    # loop over view_angles
                    for view_angle in view_angles:
                        # rot batch
                        if view_angle != 0:
                            M = cv2.getRotationMatrix2D((batch.shape[2]//2, batch.shape[1]//2), view_angle, 1)
                            for img_cnt in range(batch.shape[0]):
                                batch_rot[img_cnt] = rotate_3d(((batch[img_cnt].copy() + 0.25)*255).astype(np.uint8), M, 2)/255 - 0.25
                        else:
                            batch_rot = batch.copy()
                        # get segmentation for noduls and reshape flat prediction to batchsize, z, x, 1
                        prediction_rot = np.reshape(sess.run(pred_ops, feed_dict = {data['images']: batch_rot})['probs'], prediction.shape)                        
                        # rotate back prediction
                        if view_angle != 0:
                            M_back = cv2.getRotationMatrix2D((prediction_rot.shape[2]//2, prediction_rot.shape[1]//2), -view_angle, 1)
                            for img_cnt in range(batch.shape[0]):
                                prediction_rot[img_cnt] = np.clip(rotate_3d((prediction_rot[img_cnt]*255).astype(np.uint8), M_back, 2)/255, 0, 1)
                        # mean over view_angles
                        prediction += prediction_rot / len(view_angles)
                    # if img_shape != label_shape resizeing etc needed -> probability map has value range [0.0, 1.0]
                    if np.min(prediction) < 0 or np.max(prediction) > 1:
                         pipe.logger.error('nodule prediciton not in value range [0, 1] for patient ' + patient)
                    # crop from embedded layers
                    # print('prediction', prediction.shape)
                    prediction_embed = prediction[:, 
                                                     embed_coords_y : embed_coords_y + org_img_array.shape[0],
                                                     embed_coords_x : embed_coords_x + org_img_array.shape[1]] / len(view_planes)
                    # print('prediction_embed', prediction_embed.shape)
                    layer_start = batch_size * batch_cnt
                    # insert prob_map layer, swaping axis in right order
                    layer_end = min(img_array.shape[2], layer_start + batch_size)
                    if view_plane == 'z': # in the next line, i would like to do rollaxis(..., 2, 0)
                        # print('z', prob_map[:, :, layer_start:layer_end].shape, prediction_embed[:layer_end-layer_start, :, :, 0].shape)
                        prob_map[:, :, layer_start:layer_end] += np.swapaxes(prediction_embed[:layer_end-layer_start, :, :, 0], 0, 2) # y, x, z -> z, y, x ?!? 
                    elif view_plane == 'y': # here, i would like to do np.swapaxes(prediction_embed[:layer_end-layer_start, :, :, 0], 2, 1)
                        # print('y', prob_map[:, layer_start:layer_end, :].shape, prediction_embed[:layer_end-layer_start, :, :, 0].shape)
                        prob_map[:, layer_start:layer_end, :] += np.swapaxes(prediction_embed[:layer_end-layer_start, :, :, 0], 0, 1)  # z, x, y -> z, y, x ?!?
                    elif view_plane == 'x': # here, i would like to leave it as is, prediction_embed[:layer_end-layer_start, :, :, 0]
                        # print('x', prob_map[layer_start:layer_end, :, :].shape, prediction_embed[:layer_end-layer_start, :, :, 0].shape)
                        prob_map[layer_start:layer_end, :, :] +=  np.swapaxes(prediction_embed[:layer_end-layer_start, :, :, 0], 1, 2) # z, y, x
            # if img_shape != label_shape resizeing etc needed -> probability map has value range [0.0, 1.0]
            if np.min(prob_map) < 0 or np.max(prob_map) > 1:
                 pipe.logger.error('nodule seg prob_map not in value range [0, 1] for patient ' + patient)
            if data_type == 'uint8':
                prob_map = (prob_map * 255).astype(np.uint8)
            elif data_type == 'uint16':
                prob_map = (prob_map * 65535).astype(np.uint16)
            patients_dict[patient]['prob_map_path'] = pipe.step_dir + '/' + patient + '_prob_map.npy'
            np.save(patients_dict[patient]['prob_map_path'], prob_map)
        sess.close()
    pipe.save_step(patients_dict)

def rotate(in_tensor, M):
    dst = cv2.warpAffine(in_tensor, M, 
                         (in_tensor.shape[1], in_tensor.shape[0]), 
                         flags=cv2.INTER_CUBIC)
    if len(dst) == 2:
        dst = np.expand_dims(dst, 2)
    return dst

def rotate_3d(tensor, M, axis):
    if axis == 0:
        for i in range(tensor.shape[0]):
            tensor[i, :, :] = rotate(tensor[i, :, :], M)
    elif axis == 1:
        for i in range(tensor.shape[1]):
            tensor[:, i, :] = rotate(tensor[:, i, :], M)
    elif axis == 2:
        for i in range(tensor.shape[2]):
            tensor[:, :, i] = rotate(tensor[:, :, i], M)
    return tensor

""" Convert img-list to tensorflow TFRecord format """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import argparse
import cv2
import json
from natsort import natsorted
from tqdm import tqdm
from .. import pipeline as pipe
from .. import utils
from collections import OrderedDict

np.random.seed(21)

def run(view_angles,
        extra_radius_buffer_px,
        num_channels,
        stride,
        crop_size,
        view_planes,
        num_negative_examples_per_nodule_free_patient_per_view_plane,
        HU_tissue_range):

    if not pipe.dataset_name == 'LUNA16':
        raise ValueError('gen_nodule_seg_data only valid for dataset LUNA16')
    # dataset_dir
    # define view_planes
    view_planes = [x if x in ['y', 'x', 'z'] else 'ERROR' for x in natsorted(view_planes)]
    if len(view_planes) == 0:
        raise ValueError('No view_plane is determined!!!')
    elif 'ERROR' in view_planes:
        raise ValueError('wrong view_plane given: {}'.format(view_planes))
        sys.exit()
    if len(crop_size) != 2:
        raise ValueError('Wrong crop_size. Use format HeightxWidth')
        sys.exit()

    gen_nodule_masks_json = pipe.load_json('out.json', 'gen_nodule_masks')
    resample_lungs_json   = pipe.load_json('out.json', 'resample_lungs')

    patients_lst = OrderedDict(pipe.patients_by_split)
    for lst_type in patients_lst.keys():
        if len(patients_lst[lst_type]) == 0:
            continue
        pipe.log.info('processing lst {} with len {}'.format(lst_type, len(patients_lst[lst_type])))

        generate_data_lsts(
                HU_tissue_range,
                gen_nodule_masks_json,
                resample_lungs_json,
                lst_type,
                patients_lst[lst_type],
                num_channels,
                stride,
                crop_size,
                view_planes,
                num_negative_examples_per_nodule_free_patient_per_view_plane)


def get_slice_from_zyx_array(array, slice_start, slice_end, axis):
    slice_start = max(0, slice_start)
    slice_end   = min(slice_end, array.shape[axis])
    if slice_start>slice_end:
        print ('ERROR with slice_start, slice_end!!!')
        sys.exit()
    if axis == 0:
        return np.swapaxes(array[slice_start:slice_end].copy(),0,2)
    elif axis == 1:
        return np.swapaxes(array[:, slice_start:slice_end].copy(), 1, 2)
    elif axis == 2:
        return array[:, :, slice_start:slice_end].copy()

def ensure_point_lst_within_array(lst, array_shape):
    return [int(np.clip(x, 0, array_shape)) for x in lst]

def generate_data_lsts(HU_tissue_range,
        gen_nodule_masks_json,
        resample_lungs_json,
        lst_type, patient_lst,
        num_channels, stride,
        crop_size,
        view_planes,
        num_negatives_per_patient,
        use_shell_slices=False):
    # initialize some vars
    num_data = 0
    num_patient_data = 0
    all_data = []
    stacked_data = np.zeros(list(crop_size)+[num_channels]+[3], dtype=np.uint8)

    with open(pipe.get_step_dir() + lst_type + '_nodule_seg_data.lst', 'w') as out_lst:
        for pa_cnt, patient in enumerate(tqdm(patient_lst)):
            num_patient_data = 0

            try:
                patient_json = gen_nodule_masks_json[patient]
            except:
                raise ValueError('Could not load patient_json for patient {}'.format(patient))
                continue
            try:
                scan = pipe.load_array(resample_lungs_json[patient]['basename'], 'resample_lungs')
            except:
                raise ValueError('Could not find resample_lungs for patient {}'.format(patient))
                continue
            try:
                mask = pipe.load_array(gen_nodule_masks_json[patient]['basename'], 'gen_nodule_masks')
            except:
                raise ValueError('Could not find mask for patient {}'.format(patient))
                continue

            #normalize and zero_center scan and lab
            if scan.dtype == np.int16:
                scan = ((scan/(float(HU_tissue_range[1]-HU_tissue_range[0])))*255).astype(np.uint8)
            elif scan.dtype == np.float32:
                scan = (scan*255).astype(np.uint8)
            if mask.dtype == np.uint8:
                mask = mask

            # combine scan and mask to data
            data = np.zeros(list(scan.shape)+[3], dtype=np.uint8)
            data[:, :, :, 0]   = scan
            data[:, :, :, 1:3] = mask

            images_nodule_free = []
            nodules_extract_coords_lst = []
            nodules_center_coords_lst  = []
            images = []

            # get patient infos
            if patient_json['nodule_patient']:
                nodules = patient_json['nodules']
                for nodule in nodules:

                    #nodule_bounding_box_coords_zyx_px = nodule["nodule_box_ymin/ymax_xmin/xmax_zmin/zmax_px"]
                    factor = 1.0

                    nodule_center_zyx_px = nodule['center_zyx_px']
                    nodule_max_diameter_zyx_px = nodule['max_diameter_zyx_px']

                    nodule_bounding_box_coords_zyx_px = nodule["nodule_box_zmin/zmax_ymin/ymax_xmin/xmax_px"]

                    # ensure points within array
                    nodule_bounding_box_coords_zyx_px = [ensure_point_lst_within_array([nodule_bounding_box_coords_zyx_px[x]], data.shape[x//2])[0] for x in range(6)]
                    # ensure that bounding box has at least num_channel size
                    nodule_bounding_box_coords_zyx_px = [int(nodule_bounding_box_coords_zyx_px[v]) if v%2==0
                                                    else max(int(nodule_bounding_box_coords_zyx_px[v]), int(nodule_bounding_box_coords_zyx_px[v-1])+num_channels) for v in range(6)]
                    # get center_box_coords
                    nodule_center_box_coords_zyx_px = nodule["nodule_center_box_zmin/zmax_px_ymin/ymax_xmin/xmax"]
                    # ensure points within array
                    nodule_center_box_coords_zyx_px = [ensure_point_lst_within_array([nodule_center_box_coords_zyx_px[x]], data.shape[x//2])[0] for x in range(6)]
                    # draw center

                    # loop over view_planes
                    for view_plane in view_planes:

                        # get affected layers from scan and homogenize plan orientation (num_channels always in last dimension)
                        if view_plane =='z':
                            center_coords = [nodule_center_box_coords_zyx_px[0], nodule_center_box_coords_zyx_px[1]]
                            shell_coords  = [nodule_bounding_box_coords_zyx_px[0], nodule_bounding_box_coords_zyx_px[1]]
                            nodule_box_coords = [nodule_bounding_box_coords_zyx_px[4], nodule_bounding_box_coords_zyx_px[5], # y on first axis
                                                 nodule_bounding_box_coords_zyx_px[2], nodule_bounding_box_coords_zyx_px[3]] # x on second axis
                            axis = 0

                        elif view_plane =='y':
                            center_coords = [nodule_center_box_coords_zyx_px[2], nodule_center_box_coords_zyx_px[3]]
                            shell_coords  = [nodule_bounding_box_coords_zyx_px[2], nodule_bounding_box_coords_zyx_px[3]]
                            nodule_box_coords = [nodule_bounding_box_coords_zyx_px[0], nodule_bounding_box_coords_zyx_px[1], # z on first axis
                                                 nodule_bounding_box_coords_zyx_px[4], nodule_bounding_box_coords_zyx_px[5]] # x on second axis
                            axis = 1

                        elif view_plane =='x':
                            center_coords = [nodule_center_box_coords_zyx_px[4], nodule_center_box_coords_zyx_px[5]]
                            shell_coords  = [nodule_bounding_box_coords_zyx_px[4], nodule_bounding_box_coords_zyx_px[5]]
                            nodule_box_coords = [nodule_bounding_box_coords_zyx_px[0], nodule_bounding_box_coords_zyx_px[1], # z on first axis
                                                 nodule_bounding_box_coords_zyx_px[2], nodule_bounding_box_coords_zyx_px[3]] # y on second axis
                            axis = 2

                        if use_shell_slices:
                            shell_slices  = get_slice_from_zyx_array(data, shell_coords[0], shell_coords[1], axis=axis)
                            slices = shell_slices
                        else:
                            center_slices = get_slice_from_zyx_array(data, center_coords[0], center_coords[1], axis=axis)
                            slices = center_slices

                        num_layers = slices.shape[2]

                        # for nodules with many layers split several parts from
                        nodules_pakets = []
                        if num_layers == num_channels:
                            nodules_pakets.append(list(range(num_channels)))
                        elif num_layers > num_channels:
                            rand_offset = np.random.randint(0,int((num_layers-num_channels) % (stride))+1)
                            for paket in range(int((num_layers-num_channels)/(stride))+1):
                                nodules_pakets.append(list(np.arange(rand_offset+paket*stride, rand_offset+paket*stride+num_channels)))

                        # make nodule_pakets where label is central layer of paket
                        for nodules_paket in nodules_pakets:
                            images.append(slices[:,:,min(nodules_paket):min(nodules_paket)+num_channels])
                            nodules_extract_coords_lst.append(nodule_box_coords)

            # get some negative samples for every view_plane
            rand_layers_z = np.random.permutation(range(scan.shape[0]))
            rand_layers_y = np.random.permutation(range(scan.shape[1]))
            rand_layers_x = np.random.permutation(range(scan.shape[2]))
            rand_layer_cnt = 0
            while len(images_nodule_free) < num_negatives_per_patient*len(view_planes):
                if 'z' in view_planes:
                    idx0 = min(rand_layers_z[rand_layer_cnt], scan.shape[0]-num_channels)
                    idx1 = idx0 + num_channels
                    idx2 = np.random.randint(0, max(1,scan.shape[1]-crop_size[0]))
                    idx3 = idx2 + crop_size[0]
                    idx4 = np.random.randint(0, max(1,scan.shape[2]-crop_size[1]))
                    idx5 = idx4 + crop_size[1]
                    # introduce some random black parts
                    if np.random.randint(0,10)==0:
                        rand_black_padding = np.random.randint(0,4)
                        if rand_black_padding:
                            idx2 += np.random.randint(1,crop_size[0]//3)
                        elif rand_black_padding ==1:
                            idx3 -= np.random.randint(1,crop_size[0]//3)
                        elif rand_black_padding==2:
                            idx3 += np.random.randint(1,crop_size[1]//3)
                        elif rand_black_padding==3:
                            idx4 -= np.random.randint(1,crop_size[1]//3)

                    if np.sum(mask[idx0:idx1, idx2:idx3, idx4:idx5]) < 1:
                        images_nodule_free.append(np.swapaxes(data[idx0:idx1, idx2:idx3, idx4:idx5].copy(), 0, 2))
                        # cv2.imwrite('test_imgs/'+'y'+'_'+str(num_data)+'_'+str(len(images_nodule_free))+'_shitto.jpg', data[idx0+(idx1-idx0)//2, idx2:idx3, idx4:idx5,0])
                if 'y' in view_planes:
                    idx0 = np.random.randint(0, max(1,scan.shape[0]-crop_size[0]))
                    idx1 = idx0 + crop_size[0]
                    idx2 = min(rand_layers_y[rand_layer_cnt], scan.shape[1]-num_channels)
                    idx3 = idx2 + num_channels
                    idx4 = np.random.randint(0, max(1,scan.shape[2]-crop_size[1]))
                    idx5 = idx4 + crop_size[1]
                    # introduce some random black parts
                    if np.random.randint(0,10)==0:
                        rand_black_padding = np.random.randint(0,4)
                        if rand_black_padding:
                            idx0 += np.random.randint(1,crop_size[0]//3)
                        elif rand_black_padding ==1:
                            idx1 -= np.random.randint(1,crop_size[0]//3)
                        elif rand_black_padding==2:
                            idx3 += np.random.randint(1,crop_size[1]//3)
                        elif rand_black_padding==3:
                            idx4 -= np.random.randint(1,crop_size[1]//3)
                    if np.sum(mask[idx0:idx1, idx2:idx3, idx4:idx5]) < 1:
                        images_nodule_free.append(np.swapaxes(data[idx0:idx1, idx2:idx3, idx4:idx5].copy(), 1,2))
                        # cv2.imwrite('test_imgs/'+'x'+'_'+str(num_data)+'_'+str(len(images_nodule_free))+'_shitto.jpg', data[idx0:idx1, idx2+(idx3-idx2)//2, idx4:idx5,0])
                if 'x' in view_planes:
                    idx0 = np.random.randint(0, max(1,scan.shape[0]-crop_size[0]))
                    idx1 = idx0 + crop_size[0]
                    idx2 = np.random.randint(0, max(1,scan.shape[1]-crop_size[1]))
                    idx3 = idx2 + crop_size[1]
                    idx4 = min(rand_layers_x[rand_layer_cnt], scan.shape[2]-num_channels)
                    idx5 = idx4 + num_channels
                    # introduce some random black parts
                    if np.random.randint(0,10)==0:
                        rand_black_padding = np.random.randint(0,4)
                        if rand_black_padding:
                            idx0 += np.random.randint(1,crop_size[0]//3)
                        elif rand_black_padding ==1:
                            idx1 -= np.random.randint(1,crop_size[0]//3)
                        elif rand_black_padding==2:
                            idx2 += np.random.randint(1,crop_size[1]//3)
                        elif rand_black_padding==3:
                            idx3 -= np.random.randint(1,crop_size[1]//3)
                    if np.sum(mask[idx0:idx1, idx2:idx3, idx4:idx5]) < 1:
                        images_nodule_free.append(data[idx0:idx1, idx2:idx3, idx4:idx5].copy())
                        # cv2.imwrite('test_imgs/'+'z'+'_'+str(num_data)+'_'+str(len(images_nodule_free))+'_shitto.jpg', data[idx0:idx1, idx2:idx3, idx4+(idx5-idx4)//2,0])

                rand_layer_cnt += 1
                if rand_layer_cnt == min(len(rand_layers_z), len(rand_layers_y), len(rand_layers_x)): break

            # loop over all images and labels sprang out from nodule
            cropped_images_lsts = {'nodules': [images, nodules_extract_coords_lst], 'nodule_free': [images_nodule_free, [None]*len(images_nodule_free)]}

            for cropped_images_lst_key in cropped_images_lsts.keys():
                zipped_lsts = cropped_images_lsts[cropped_images_lst_key]
                for img_cnt in range(len(zipped_lsts[0])):

                    org_img = zipped_lsts[0][img_cnt][:,:,:,:1]
                    org_lab = zipped_lsts[0][img_cnt][:,:,:,1:3]

                    nodule_box_coords_in_extract = zipped_lsts[1][img_cnt]

                    img = np.zeros(crop_size+[num_channels]+[1], dtype=np.uint8)
                    lab = np.zeros(crop_size+[1]+[2], dtype=np.uint8) # first channel ist label to predict, second channel for drawing center

                    # crop or pad org_img
                    img_coords = []
                    org_coords = []
                    if patient_json['nodule_patient']:
                        # crop random around nodule or pad black

                        # ensure that nodule is within crop
                        for idx in range(2):
                            if crop_size[idx] < org_img.shape[idx]:
                                if patient_json['nodule_patient']:
                                    img_coords.append(0)
                                    img_coords.append(crop_size[idx])
                                    if (nodule_box_coords_in_extract[idx*2+1]-nodule_box_coords_in_extract[idx*2])<crop_size[idx]:
                                        start_random = max(0,min(org_img.shape[idx]-crop_size[idx], nodule_box_coords_in_extract[idx*2+1]-(crop_size[idx])))
                                        end_random   = max(start_random+1, min(nodule_box_coords_in_extract[idx*2], org_img.shape[idx]-crop_size[idx]))
                                        org_coords.append(np.random.randint(start_random, end_random))
                                        org_coords.append(org_coords[-1]+crop_size[idx])
                                    else:
                                        org_coords.append(np.random.randint(0, org_img.shape[idx]-crop_size[idx]))
                                        org_coords.append(org_coords[-1]+crop_size[idx])
                                else:
                                    img_coords.append(0)
                                    img_coords.append(crop_size[idx])
                                    org_coords.append(np.random.randint(0, org_img.shape[idx]-crop_size[idx]))
                                    org_coords.append(org_coords[-1]+crop_size[idx])

                            elif crop_size[idx] >= org_img.shape[idx]:
                                img_coords.append((crop_size[idx]-org_img.shape[idx])/2)
                                img_coords.append(img_coords[-1]+org_img.shape[idx])
                                org_coords.append(0)
                                org_coords.append(org_img.shape[idx])

                    else:
                        # crop or pad negative_img
                        for idx in range(2):
                            if org_img.shape[idx] >= img.shape[idx]:
                                # start
                                img_coords.append(0)
                                org_coords.append((org_img.shape[idx]-img.shape[idx])//2)
                                # end
                                img_coords.append(img.shape[idx])
                                org_coords.append(org_coords[-1]+img.shape[idx])
                            else:
                                # start
                                img_coords.append((img.shape[idx]-org_img.shape[idx])//2)
                                org_coords.append(0)
                                # end
                                img_coords.append(img_coords[-1]+org_img.shape[idx])
                                org_coords.append(org_img.shape[idx])

                    img_coords = [int(x) for x in img_coords]
                    org_coords = [int(x) for x in org_coords]
                    img[img_coords[0]:img_coords[1], img_coords[2]:img_coords[3], :] = org_img[org_coords[0]:org_coords[1], org_coords[2]:org_coords[3]].copy()
                    lab[img_coords[0]:img_coords[1], img_coords[2]:img_coords[3], :] = org_lab[org_coords[0]:org_coords[1], org_coords[2]:org_coords[3]].copy()

                    stacked_data[:,:,:img.shape[-1],:1]  = img.copy()
                    stacked_data[:,:,:lab.shape[-1],1:3] = lab.copy()
                    if 0 and cropped_images_lst_key=='nodules':
                        randy = np.random.randint(0,10000)
                        cv2.imwrite('test_imgs/'+str(randy)+'_img.jpg', stacked_data[:,:,stacked_data.shape[2]//2,0])
                        cv2.imwrite('test_imgs/'+str(randy)+'_lab.jpg', stacked_data[:,:,stacked_data.shape[2]//2,1])
                        cv2.imwrite('test_imgs/'+str(randy)+'_center.jpg', stacked_data[:,:,stacked_data.shape[2]//2,2])
                    all_data.append(stacked_data.copy())

                    out_lst.write('{}\t{}\t{}\t{}\n'.format(num_data, patient, num_patient_data, 1 if cropped_images_lst_key=='nodules' else 0))
                    num_data += 1
                    num_patient_data += 1
                    # if num_data == 10000 and 0: 
                    #     np.save(out_folder_path+out_lst_name+'.npy', np.array(all_data))
                    #     out_lst.close()
                    #     sys.exit()

    print ('---------------NUM_DATA: {}-------------------'.format(num_data))
    print ('saving npy-construct')
    out_path = pipe.save_array(basename=lst_type+'.npy' , array=np.array(all_data), step_name='gen_nodule_seg_data')
    print ('saved npy for list {} to {}. Its shape is {}.'.format( lst_type, out_path, np.array(all_data).shape))

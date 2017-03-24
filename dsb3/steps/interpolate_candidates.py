"""
Interpolate low-resolution candidates to high resolution.
"""
import os
import sys
import numpy as np
import pandas as pd
import cv2
import json
from tqdm import tqdm
from collections import OrderedDict
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from . import resample_lungs
from .. import pipeline as pipe


def run(n_candidates,
        new_spacing_zyx,
        new_candidates_shape_zyx,
        new_data_type,
        crop_raw_scan_buffer):
    avail_data_types = ['uint8', 'int16', 'float32']
    if new_data_type not in avail_data_types:
        raise ValueError('Wrong data type, choose one of ' + str(avail_data_types))
    gen_candidates_json = pipe.load_json('out.json', 'gen_candidates')
    resample_lungs_json = pipe.load_json('out.json', 'resample_lungs')
    input_lst = pd.read_csv(pipe.get_step_dir('gen_candidates') + 'patients.lst', sep = '\t', header=None)
    img_lsts_dict = OrderedDict()
    img_candidates_lsts_dict = OrderedDict()
    for split_name, split in pipe.patients_by_split.items():
        img_lsts_dict[split_name] = input_lst[input_lst[0].isin(split)]
    if pipe.dataset_name == 'LUNA16':
        input_lst_candidates = pd.read_csv(pipe.get_step_dir('gen_candidates') + 'candidates.lst', sep = '\t', header=None)
        for split_name, split in pipe.patients_by_split.items():
            truncated_first_column = pd.Series([name.split('_')[0] for name in input_lst_candidates[0]])
            img_candidates_lsts_dict[split_name] = input_lst_candidates[truncated_first_column.isin(split)]
    for lst_type in img_lsts_dict.keys():
        pipe.log.info('processing lst {} with len {}'.format(lst_type, len(img_lsts_dict[lst_type])))
        if len(img_lsts_dict[lst_type]) == 0:
            continue
        img_lst_patients = img_lsts_dict[lst_type]
        if img_candidates_lsts_dict is not None:
            img_lst_candidates = img_candidates_lsts_dict[lst_type]
        else:
            img_lst_candidates = None
        # ensure output files are overwritten
        open(pipe.get_step_dir() + lst_type + '_patients.lst', 'w').close()
        if pipe.dataset_name == 'LUNA16':
            open(pipe.get_step_dir() + lst_type + '_candidates.lst', 'w').close()
        gen_data(lst_type,
                 img_lst_patients,
                 img_lst_candidates,
                 gen_candidates_json,
                 resample_lungs_json,
                 n_candidates,
                 crop_raw_scan_buffer,
                 new_data_type,
                 new_candidates_shape_zyx,
                 new_spacing_zyx)

def gen_data(lst_type,
             img_lst_patients,
             img_lst_candidates,
             gen_candidates_json,
             resample_lungs_json,
             n_candidates,
             crop_raw_scan_buffer,
             new_data_type,
             new_candidates_shape_zyx,
             new_spacing_zyx):
    n_threads = pipe.n_CPUs
    n_junks = int(np.ceil(len(img_lst_patients) / n_threads))
    pipe.log.info('processing ' + str(n_junks) + ' junks with ' + str(n_threads) + ' patients each')
    HU_tissue_range = pipe.load_json('params.json', 'resample_lungs')['HU_tissue_range']
    n_candidates_gen = pipe.load_json('params.json', 'gen_candidates')['n_candidates']
    cand_line_num = 0
    for junk_cnt in range(n_junks):
        junk = []
        for in_junk_cnt in range(n_threads):
            line_num = n_threads * junk_cnt + in_junk_cnt
            if line_num >= len(img_lst_patients):
                break
            junk.append(line_num)
        pipe.log.info('processing junk ' + str(junk_cnt))
        # heterogenous spacing -> homogeneous spacing
        junk_lst = Parallel(n_jobs=min([n_threads, len(junk)]))(
                            delayed(gen_patients_candidates)(line_num,
                                                             img_lst_patients,
                                                             gen_candidates_json,
                                                             resample_lungs_json,
                                                             n_candidates,
                                                             crop_raw_scan_buffer,
                                                             new_data_type,
                                                             new_candidates_shape_zyx,
                                                             new_spacing_zyx) for line_num in junk)
        for junk_result in junk_lst:
            patient, patient_label, images, prob_maps = junk_result
            images = np.array(images, dtype=np.int16)
            prob_maps = np.array(prob_maps, dtype=np.uint8)
            # fill in zeros if there are less than n_candidates in the array
            if images.shape[0] < n_candidates:
                images = np.vstack((images, np.zeros([n_candidates - images.shape[0]] + list(images.shape[1:]), dtype='int16')))
                prob_maps = np.vstack((prob_maps, np.zeros([n_candidates - prob_maps.shape[0]] + list(prob_maps.shape[1:]), dtype='int16')))
            if new_data_type == 'uint8':
                images = (images / (float(HU_tissue_range[1] - HU_tissue_range[0])) * 255).astype(np.uint8) # [0, 255]
            elif new_data_type == 'float32':
                images = (images / (float(HU_tissue_range[1] - HU_tissue_range[0])) - 0.25).astype(np.float32) # [-0.25, 0.75]
                prob_maps = (prob_maps / 255).astype(np.float32) # [0.0, 1.0]
            images_and_prob_maps = np.concatenate([images, prob_maps], axis=4).astype(new_data_type)
            path = pipe.save_array(patient + '.npy', images_and_prob_maps)
            with open(pipe.get_step_dir() + lst_type + '_patients.lst', 'a') as f:
                f.write('{}\t{}\t{}\n'.format(patient, patient_label, path))
            if pipe.dataset_name == 'LUNA16':
                with open(pipe.get_step_dir() + lst_type + '_candidates.lst', 'a') as f:
                    for cnt in range(len(images)):
                        cand = img_lst_candidates[0][cand_line_num]
                        cand_label = img_lst_candidates[1][cand_line_num]
                        if not cand.startswith(patient):
                            raise ValueError(cand + ' needs to start with ' + patient)
                        path = pipe.save_array(cand + '.npy', images_and_prob_maps[cnt])
                        f.write('{}\t{}\t{}\n'.format(cand, cand_label, path))
                        cand_line_num += 1
            cand_line_num += n_candidates_gen - n_candidates

def gen_patients_candidates(line_num,
                            img_lst_patients,
                            gen_candidates_json,
                            resample_lungs_json,
                            n_candidates,
                            crop_raw_scan_buffer,
                            new_data_type,
                            new_candidates_shape_zyx,
                            new_spacing_zyx):
    patient = img_lst_patients[0][line_num]
    patient_label = img_lst_patients[1][line_num]
    lung_box_coords_zyx_px = [0, 0] + resample_lungs_json[patient]['bound_box_coords_yx_px'] # offset from lung_wings
    lung_box_offset_zzyyxx_px = [lung_box_coords_zyx_px[i // 2] for i in range(6)]
    raw_scan_spacing_zyx_mm_px = resample_lungs_json[patient]['raw_scan_spacing_zyx_mm']
    resampled_scan_spacing_zyx_mm_px = resample_lungs_json[patient]['resampled_scan_spacing_zyx_mm']
    HU_tissue_range = resample_lungs_json['HU_tissue_range']
    convert2raw_scan_spacing_factor = np.array(resampled_scan_spacing_zyx_mm_px) / np.array(raw_scan_spacing_zyx_mm_px, dtype=np.float32) #zyx
    convert2raw_scan_spacing_factor = [convert2raw_scan_spacing_factor[j] for j in range(3) for i in range(2)] #zzyyxx
    if pipe.dataset_name == 'LUNA16':
        lung_array_zyx, old_spacing_zyx, _, _ = resample_lungs.get_img_array_mhd(pipe.patients_raw_data_paths[patient])
    elif pipe.dataset_name == 'dsb3':
        lung_array_zyx, old_spacing_zyx, _, _ = resample_lungs.get_img_array_dcom(pipe.patients_raw_data_paths[patient])
    clusters = gen_candidates_json[patient]['clusters']
    images = []; prob_maps = []
    for clu_num, clu in enumerate(clusters[:min(len(clusters), n_candidates)]):
        candidate_box_coords_zyx_px = np.array(clu['candidate_box_coords_zyx_px']) + np.array(lung_box_offset_zzyyxx_px)
        candidate_box_coords_raw_zyx_px = [((candidate_box_coords_zyx_px[dim] - crop_raw_scan_buffer) 
                                             if dim % 2 == 0 else
                                            (candidate_box_coords_zyx_px[dim] + crop_raw_scan_buffer)) * convert2raw_scan_spacing_factor[dim]
                                             for dim in range(6)]
        # '+ 1': convention is that box coords contain candidate but don't go beyond, see also gen_nodules_masks
        z_start = int(max(0, candidate_box_coords_raw_zyx_px[0]))
        z_end   = int(min(candidate_box_coords_raw_zyx_px[1] + 1, lung_array_zyx.shape[0]))
        y_start = int(max(0, candidate_box_coords_raw_zyx_px[2]))
        y_end   = int(min(candidate_box_coords_raw_zyx_px[3] + 1, lung_array_zyx.shape[1]))
        x_start = int(max(0, candidate_box_coords_raw_zyx_px[4]))
        x_end   = int(min(candidate_box_coords_raw_zyx_px[5] + 1, lung_array_zyx.shape[2]))
        image = lung_array_zyx[z_start:z_end, y_start:y_end, x_start:x_end].copy()
        image = resample_lungs.resize_and_interpolate_array(image, old_spacing_zyx, new_spacing_zyx)
        image = resample_lungs.clip_HU_range(image, HU_tissue_range)
        image_box = np.zeros(new_candidates_shape_zyx, dtype='int16')
        # offset_z = int((image_box.shape[0] - image.shape[0])/2)
        # offset_y = int((image_box.shape[1] - image.shape[1])/2)
        # offset_x = int((image_box.shape[2] - image.shape[2])/2)
        # image_box[offset_z : offset_z + image.shape[0],
        #           offset_y : offset_y + image.shape[1],
        #           offset_x : offset_x + image.shape[2]] = image
        image = image[:new_candidates_shape_zyx[0],
                      :new_candidates_shape_zyx[1],
                      :new_candidates_shape_zyx[2]]
        # load low-resolution prob_map and resize to high-resolution
        # 'int16': account for that interpolation that might induce values below 0 or above 255
        prob_map = pipe.load_array(clu['prob_map_basename'], 'gen_candidates').astype('int16') # z, y, x
        prob_map = resample_lungs.resize_and_interpolate_array(prob_map, resampled_scan_spacing_zyx_mm_px, new_spacing_zyx)
        prob_map = np.clip(prob_map, 0, 255).astype('uint8') # back to uint8
        prob_map = prob_map[:new_candidates_shape_zyx[0],
                            :new_candidates_shape_zyx[1],
                            :new_candidates_shape_zyx[2]]
        # visualize
        if np.random.randint(0, 1) == 0:
            old_image = pipe.load_array(clu['img_basename'], 'gen_candidates')
            plt.imshow(old_image[old_image.shape[0] // 2, :, :])
            plt.savefig(pipe.get_step_dir() + 'figs/' + patient + '_can' + str(clu_num) + '_imgold.png')
            plt.clf()
            plt.imshow(image[image.shape[0] // 2, :, :])
            plt.savefig(pipe.get_step_dir() + 'figs/' + patient + '_can' + str(clu_num) + '_imgnew.png')
            plt.clf()
            old_prob_map = pipe.load_array(clu['prob_map_basename'], 'gen_candidates')
            plt.imshow(old_prob_map[old_prob_map.shape[0] // 2, :, :])
            plt.savefig(pipe.get_step_dir() + 'figs/' + patient + '_can' + str(clu_num) + '_probold.png')
            plt.clf()
            plt.imshow(prob_map[prob_map.shape[0] // 2, :, :])
            plt.savefig(pipe.get_step_dir() + 'figs/' + patient + '_can' + str(clu_num) + '_probnew.png')
            plt.clf()
        # expand dimensions
        image = np.expand_dims(image, 3)
        images.append(image)
        prob_map = np.expand_dims(prob_map, 3)
        prob_maps.append(prob_map)
    return [patient, patient_label, images, prob_maps]

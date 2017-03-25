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
from .. import utils

def run(n_candidates,
        new_spacing_zyx,
        new_candidates_shape_zyx,
        new_data_type,
        crop_raw_scan_buffer):
    """
    n_candidates : int
        Number of candidates to interpolate. Should be lower or as high as the 
        number of low resolution candidates produced in `gen_candidates`.
    new_spacing_zyx : tuple of len 3
        Spacing as (z, y, x).
    new_candidates_shape_zyx : tuple of len 3
        For example (96, 96, 96).
    new_data_type : {'uint8', 'int16', 'float32'}
    crop_raw_scan_buffer : int
        Number of pixels to add to raw scan buffer.
    """
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
        img_lsts_dict[split_name].reset_index(drop=True, inplace=True)
    if pipe.dataset_name == 'LUNA16':
        input_lst_candidates = pd.read_csv(pipe.get_step_dir('gen_candidates') + 'candidates.lst', sep = '\t', header=None)
        for split_name, split in pipe.patients_by_split.items():
            truncated_first_column = pd.Series([name.split('_')[0] for name in input_lst_candidates[0]])
            img_candidates_lsts_dict[split_name] = input_lst_candidates[truncated_first_column.isin(split)]
            img_candidates_lsts_dict[split_name].reset_index(drop=True, inplace=True)
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
                                                             new_spacing_zyx,
                                                             HU_tissue_range) for line_num in junk)
        for junk_result in junk_lst:
            patient, patient_label, images, prob_maps = junk_result
            # take n_candidates or less
            images = np.array(images, dtype=np.int16)[:n_candidates]
            prob_maps = np.array(prob_maps, dtype=np.uint8)[:n_candidates]
            # get num real (not filled) candidates
            num_real_candidates = images.shape[0]
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
                    for cnt in range(num_real_candidates):
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
                            new_spacing_zyx,
                            HU_tissue_range):
    patient = img_lst_patients[0][line_num]
    patient_label = img_lst_patients[1][line_num]
    lung_box_coords_zyx_px = [0, 0] + resample_lungs_json[patient]['bound_box_coords_yx_px'] # offset from lung_wings
    lung_box_offset_zzyyxx_px = [lung_box_coords_zyx_px[2*j] for j in range(3) for i in range(2)]
    raw_scan_spacing_zyx_mm_px = resample_lungs_json[patient]['raw_scan_spacing_zyx_mm']
    resampled_scan_spacing_zyx_mm_px = resample_lungs_json[patient]['resampled_scan_spacing_zyx_mm']
    convert2raw_scan_spacing_factor = np.array(resampled_scan_spacing_zyx_mm_px, dtype='float32') / np.array(raw_scan_spacing_zyx_mm_px) #zyx
    convert2raw_scan_spacing_factor = [convert2raw_scan_spacing_factor[j] for j in range(3) for i in range(2)] #zzyyxx
    if pipe.dataset_name == 'LUNA16':
        raw_lung_array, old_spacing_zyx, _, _ = resample_lungs.get_img_array_mhd(pipe.patients_raw_data_paths[patient])
    elif pipe.dataset_name == 'dsb3':
        raw_lung_array, old_spacing_zyx, _, _ = resample_lungs.get_img_array_dcom(pipe.patients_raw_data_paths[patient])
    clusters = gen_candidates_json[patient]['clusters']
    images = []; prob_maps = []
    for clu_num, clu in enumerate(clusters[:min(len(clusters), n_candidates)]):
        candidate_box_coords_zyx_px = list(np.array(clu['box_coords_px']) + np.array(lung_box_offset_zzyyxx_px))
        crop_raw = [int(np.round(candidate_box_coords_zyx_px[dim] * convert2raw_scan_spacing_factor[dim])) for dim in range(6)]
        crop_raw = [int(max(0, crop_raw[i] - 1)) # tiny buffer here
                    if i % 2 == 0 else 
                    int(min(crop_raw[i] + 1, raw_lung_array.shape[i // 2])) for i in range(6)]
        image_raw = raw_lung_array[crop_raw[0]:crop_raw[1], crop_raw[2]:crop_raw[3], crop_raw[4]:crop_raw[5]].copy()
        image = resample_lungs.resize_and_interpolate_array(image_raw, old_spacing_zyx, new_spacing_zyx)
        image = resample_lungs.clip_HU_range(image, HU_tissue_range)
        # consider accounting for an offset as in gen_prob_maps
        new_shape = new_candidates_shape_zyx
        new_box = [int((image.shape[i // 2] - new_shape[i // 2])/2) if i %2 == 0 
                   else int((image.shape[i // 2] - new_shape[i // 2])/2) + new_shape[i // 2] for i in range(6)]
        image = utils.crop_and_embed(image, new_box, new_shape)
        # 'int16': account for that interpolation that might induce values below 0 or above 255
        prob_map = pipe.load_array(clu['prob_map_basename'], 'gen_candidates').astype('int16') # z, y, x
        prob_map = resample_lungs.resize_and_interpolate_array(prob_map, resampled_scan_spacing_zyx_mm_px, new_spacing_zyx)
        prob_map = np.clip(prob_map, 0, 255).astype('uint8') # back to uint8
        prob_map = prob_map[:new_candidates_shape_zyx[0],
                            :new_candidates_shape_zyx[1],
                            :new_candidates_shape_zyx[2]]
        # visualize
        if np.random.randint(0, 100) == 0:
            old_image = pipe.load_array(clu['img_basename'], 'gen_candidates')
            plt.imshow(old_image[old_image.shape[0] // 2, :, :])
            plt.savefig(pipe.get_step_dir() + 'figs/' + patient + '_can' + str(clu_num) + '_imgold.png')
            plt.clf()
            plt.imshow(image[image.shape[0] // 2, :, :])
            plt.savefig(pipe.get_step_dir() + 'figs/' + patient + '_can' + str(clu_num) + '_imgnew.png')
            plt.clf()
            plt.imshow(image_raw[image_raw.shape[0] // 2, :, :])
            plt.savefig(pipe.get_step_dir() + 'figs/' + patient + '_can' + str(clu_num) + '_imgraw.png')
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

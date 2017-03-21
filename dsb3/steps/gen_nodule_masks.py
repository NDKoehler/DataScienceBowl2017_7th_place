from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os,sys
import cv2
import scipy.ndimage
from natsort import natsorted
import json
import argparse
from master_config import H
from utils import ensure_dir
from ellipse_helpers import *
import matplotlib.pyplot as plt
import code
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(17) # do NOT change
nodule_patients_set = set() # monitor parallel processing
nodule_patients_processed_set = set() # monitor parallel processing

def get_img_array(resampled_lung_path):
    resample_lung = np.load(resampled_lung_path)
    return resample_lung

def draw_ellipse(mask_array, color, v_center_px, v_diam_px):
    v_diam_px = 2*v_diam_px
    X = np.argwhere(mask_array > 1)
    center, radii, rotation = getMinVolEllipse(X, tolerance=0.01, v_center_px=v_center_px, v_diam_px=v_diam_px)
    new_mask_shell                    = np.zeros_like(mask_array, dtype=np.uint8)
    new_mask_shell, yxz_bbox_px_shell = draw_new_ellipsoid(new_mask_shell, center, [max(H['gen_nodules_mask2pred_lower_radius_limit_px'],int(r)) for r in radii], rotation, v_center_px, v_diam_px, color)
    # draw second mask with reduced size
    new_mask_center                     = np.zeros_like(mask_array, dtype=np.uint8)
    new_mask_center, yxz_bbox_px_center = draw_new_ellipsoid(new_mask_center, center, [max(1,int(r*float(H['gen_nodules_reduced_mask_radius_fraction']))) for r in radii], rotation, v_center_px, v_diam_px, color)

    # print(np.unique(new_mask, return_counts=True))
    # plot__both_scatters(new_points, X)
    return new_mask_shell, yxz_bbox_px_shell, new_mask_center, yxz_bbox_px_center

def draw_new_ellipsoid(new_mask, center, radii, rotation, v_center_px, v_diam_px, color):
    for i in range(max(0, v_center_px[0]-int(round(v_diam_px[0]))-15), min(v_center_px[0]+int(round(v_diam_px[0]))+15, new_mask.shape[0]-1)):
        for j in range(max(0, v_center_px[1]-int(round(v_diam_px[1]))-15), min(v_center_px[1]+int(round(v_diam_px[1]))+15, new_mask.shape[1]-1)):
            for k in range(max(0, v_center_px[2]-int(round(v_diam_px[2]))-15), min(v_center_px[2]+int(round(v_diam_px[2]))+15, new_mask.shape[2]-1)):
                r = np.array([i,j,k])
                r = np.dot(rotation, r - center)
                c_value = (r[0]/radii[0])**2 + (r[1]/radii[1])**2 + (r[2]/radii[2])**2 # Ellipse constraint
                if c_value <=1:
                    new_mask[i,j,k]=color
    new_points = np.argwhere(new_mask> 1)
    yxz_bbox_px = [np.min(new_points[:,0]), np.max(new_points[:,0]), np.min(new_points[:,1]), np.max(new_points[:,1]), np.min(new_points[:,2]), np.max(new_points[:,2])]
    return new_mask, yxz_bbox_px

def plot(mask, image):
    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()

def make_nodule(nodule_annotations, mask_array, img_array, origin, real_spacing, lung_bounding_box_offset_yxz_px, yx_buffer_px, z_buffer_px, patient, limit_px=H['gen_nodules_mask2pred_upper_radius_limit_px']):
    new_mask_array = np.zeros_like(mask_array,dtype=np.uint8)
    new_mask_array_shell  = new_mask_array[:,:,:,0]
    new_mask_array_center = new_mask_array[:,:,:,1]
    nodule_annotations.sort_values("coordZ")
    # converting coordinates to pixels
    nodule_annotations["coordX_px"] = ((nodule_annotations["coordX"].copy() - origin[1])/real_spacing[1] - lung_bounding_box_offset_yxz_px[1]).round(decimals=0).astype(int)
    nodule_annotations["coordY_px"] = ((nodule_annotations["coordY"].copy() - origin[0])/real_spacing[0] - lung_bounding_box_offset_yxz_px[0]).round(decimals=0).astype(int)
    nodule_annotations["coordZ_px"] = ((nodule_annotations["coordZ"].copy() - origin[2])/real_spacing[2] - lung_bounding_box_offset_yxz_px[2]).round(decimals=0).astype(int)
    nodule_annotations["diameter_x_px"] = [min(va/real_spacing[1]  + yx_buffer_px, limit_px) for va in nodule_annotations["diameter_x_mm"]]
    nodule_annotations["diameter_y_px"] = [min(va/real_spacing[1]  + yx_buffer_px, limit_px) for va in nodule_annotations["diameter_y_mm"]]
    real_center_mm = [np.mean(nodule_annotations["coordY"]), np.mean(nodule_annotations["coordX"]), np.mean(nodule_annotations["coordZ"])]
    # no buffer
    z_min_px = int(np.ceil((nodule_annotations["z_min_mm"].iloc[0] - origin[2])/real_spacing[2]))
    z_max_px = int(np.floor((nodule_annotations["z_max_mm"].iloc[0] - origin[2])/real_spacing[2]))
    # limit size to max 24 px
    z_min_px = max(z_min_px+(z_max_px-z_min_px)//2 - limit_px//2, z_min_px)
    z_max_px = min(z_max_px-(z_max_px-z_min_px)//2 + limit_px//2, z_max_px)
    # [y, x, z]
    v_center_px = [int(round(np.mean(nodule_annotations["coordY_px"]))), int(round(np.mean(nodule_annotations["coordX_px"]))), int(round(np.mean(nodule_annotations["coordZ_px"])))]
    # only used for json
    v_diam_px = [np.max(nodule_annotations["diameter_y_mm"]/real_spacing[0]), np.max(nodule_annotations["diameter_x_mm"]/real_spacing[1]), np.abs(z_max_px-z_min_px)]
    v_diam_px = [max(2,v) for v in v_diam_px]
    old_diameter_mm = np.max(nodule_annotations["diameter_mm"])
    nodule_priority = nodule_annotations["nodule_priority"].iloc[0]
    if nodule_priority >= 3:
        color = 255
    elif nodule_priority == 2:
        color = 170
    elif nodule_priority == 1:
        color = 85
    else:
        print ('wrong nodule_priority!!! in')
        sys.exit()
    start_layer = max(0, z_min_px - z_buffer_px)
    end_layer = min(new_mask_array[:,:,:,0].shape[2]-1, (z_max_px + z_buffer_px))
    affected_layers = list(np.arange(start_layer, end_layer+1,dtype=int))

    # avoid memory overflow
    if v_diam_px[0]*v_diam_px[1]*v_diam_px[2] <= 1000:
        thickness = -1
    else:
        thickness = 2

    for idx, v_layer in enumerate(affected_layers):
        v_layer = int(v_layer)
        # in y, x, z
        rad_dec = 1# decrease radii around z factor
        # Case1: z has its own x_rad and y_rad
        # middle = int(round((z_max_px - z_min_px)/2 + z_min_px))
        if z_min_px <= v_layer <= z_max_px:
            # print("middle")
            # position of the v_layer in mm
            CoordZ_mm = (v_layer + lung_bounding_box_offset_yxz_px[2])*real_spacing[2] + origin[2]
            # taking the annotation that is closest to the layer´s z-position
            x_rad = nodule_annotations["diameter_x_px"].iloc[(nodule_annotations["coordZ"]-CoordZ_mm).abs().argsort().iloc[0]]//2
            y_rad = nodule_annotations["diameter_y_px"].iloc[(nodule_annotations["coordZ"]-CoordZ_mm).abs().argsort().iloc[0]]//2
            x_center = nodule_annotations["coordX_px"].iloc[(nodule_annotations["coordZ"]-CoordZ_mm).abs().argsort().iloc[0]]
            y_center = nodule_annotations["coordY_px"].iloc[(nodule_annotations["coordZ"]-CoordZ_mm).abs().argsort().iloc[0]]
            ellipse_radi_xy = tuple([int(y_rad), int(x_rad)])
        # Case2: z does not have its x_rad and y_rad, due to buffering
        elif v_layer < z_min_px:
            # print("below")
            x_rad = nodule_annotations["diameter_x_px"].ix[nodule_annotations["coordZ"].idxmin()]//2
            y_rad = nodule_annotations["diameter_y_px"].ix[nodule_annotations["coordZ"].idxmin()]//2
            x_center = nodule_annotations["coordX_px"].ix[nodule_annotations["coordZ"].idxmin()]
            y_center = nodule_annotations["coordY_px"].ix[nodule_annotations["coordZ"].idxmin()]
            ellipse_radi_xy = tuple([int(round(np.sqrt(max(1, x_rad**2-(rad_dec*(z_min_px-v_layer))**2)))), int(round(np.sqrt(max(1, y_rad**2-(rad_dec*(z_min_px-v_layer))**2))))])
        elif v_layer > z_max_px:
            # print("above")
            x_rad = nodule_annotations["diameter_x_px"].ix[nodule_annotations["coordZ"].idxmax()]//2
            y_rad = nodule_annotations["diameter_y_px"].ix[nodule_annotations["coordZ"].idxmax()]//2
            x_center = nodule_annotations["coordX_px"].ix[nodule_annotations["coordZ"].idxmax()]
            y_center = nodule_annotations["coordY_px"].ix[nodule_annotations["coordZ"].idxmax()]
            ellipse_radi_xy = tuple([int(round(np.sqrt(max(1, x_rad**2-(rad_dec*(z_max_px-v_layer))**2)))), int(round(np.sqrt(max(1, y_rad**2-(rad_dec*(z_max_px-v_layer))**2))))])

        ellipse_radi_xy = tuple([max(2,v) for v in ellipse_radi_xy])
        layer_to_draw_in = new_mask_array_shell[:,:,v_layer].copy()
        cv2.ellipse(layer_to_draw_in, center=tuple([x_center, y_center]), axes=ellipse_radi_xy, angle=0, startAngle=0, endAngle=360, color=(color), thickness=thickness)
        new_mask_array_shell[:,:,v_layer] = np.clip(layer_to_draw_in + new_mask_array_shell[:,:,v_layer],0, 255)

    new_mask_array_shell, yxz_bbox_px_shell, new_mask_array_center, yxz_bbox_px_center   = draw_ellipse(new_mask_array_shell, color, v_center_px, v_diam_px)
    # Ensure nodule priority in case of overlap - always take maximum
    mask_array[:,:,:,0] = np.maximum(new_mask_array_shell,  mask_array[:,:,:,0])
    mask_array[:,:,:,1] = np.maximum(new_mask_array_center, mask_array[:,:,:,1])
    # draw center in center channel
    if 0:
        randy = np.random.randint(0,1000)
        for cnt, z_layer in enumerate(np.arange(v_center_px[2] - 2, v_center_px[2] + 2)):
            cv2.imwrite('test_imgs/'+patient+ '_'+str(randy)+'_center_'+str(cnt)+'.jpg', mask_array[:,:,z_layer,1])
            cv2.imwrite('test_imgs/'+patient+ '_'+str(randy)+'_img_'+str(cnt)+'.jpg', (255/1400.*img_array[:,:,z_layer]).astype(np.uint8))
            cv2.imwrite('test_imgs/'+patient+ '_'+str(randy)+'_shell_'+str(cnt)+'.jpg', mask_array[:,:,z_layer,0])
    return mask_array, v_center_px, real_center_mm, v_diam_px, old_diameter_mm, yxz_bbox_px_shell, yxz_bbox_px_center

def save_nodule_mask(mask_array, folder_name, patient):
    path_to_mask = dataset_dir + folder_name+'/'+patient+"_mask.npy"
    np.save(path_to_mask, mask_array)
    return path_to_mask

def process_nodule_patient(patient, patient_json):
    patient_annotation = annotations[annotations['seriesuid']==str(patient)]
    patient_json["nodule_patient"]=True
    patient_json['number_of_nodules']=len(set(patient_annotation["nodule_id"]))
    patient_json['nodules']=[]
    img_array = get_img_array(patient_json['resampled_lung_path'])
    lung_bounding_box_offset_yxz_px = patient_json['lung_bounding_cube_coords_yx_px'] # no offset in z direction
    lung_bounding_box_offset_yxz_px = [lung_bounding_box_offset_yxz_px[0], lung_bounding_box_offset_yxz_px[2],0]
    real_spacing = patient_json['resampled_scan_spacing_yxz_mm/px']
    raw_spacing = patient_json['raw_scan_spacing_yxz_mm/px']
    origin = patient_json['origin_yxz_mm']
    mask_array = np.zeros(list(img_array.shape)+[2],dtype=np.uint8)
    for nodule_id in set(patient_annotation["nodule_id"]):
        nodule_annotations = patient_annotation[patient_annotation["nodule_id"]==nodule_id]
        mask_array, v_center_px, real_center_mm, v_diam_px, old_diameter_mm, yxz_bbox_px, center_box_coords_yxz_px = make_nodule(nodule_annotations, mask_array, img_array.copy(), origin, real_spacing, lung_bounding_box_offset_yxz_px, yx_buffer_px, z_buffer_px, patient)
        y_min_bbox_mm = (yxz_bbox_px[0]+lung_bounding_box_offset_yxz_px[0])*real_spacing[0]+origin[0]
        y_max_bbox_mm = (yxz_bbox_px[1]+lung_bounding_box_offset_yxz_px[0])*real_spacing[0]+origin[0]
        x_min_bbox_mm = (yxz_bbox_px[2]+lung_bounding_box_offset_yxz_px[1])*real_spacing[1]+origin[1]
        x_max_bbox_mm = (yxz_bbox_px[3]+lung_bounding_box_offset_yxz_px[1])*real_spacing[1]+origin[1]
        z_min_bbox_mm = (yxz_bbox_px[4]+lung_bounding_box_offset_yxz_px[2])*real_spacing[2]+origin[2]
        z_max_bbox_mm = (yxz_bbox_px[5]+lung_bounding_box_offset_yxz_px[2])*real_spacing[2]+origin[2]
        yxz_bbox_mm = [y_min_bbox_mm, y_max_bbox_mm, x_min_bbox_mm, x_max_bbox_mm, z_min_bbox_mm, z_max_bbox_mm]

        # Look at the center of the annotation
        # plt.imshow(mask_array[:,:,v_center_px[2]])
        # plt.show()
        # plt.imshow(img_array[:,:,v_center_px[2]]+0.25)
        # plt.show()
        nodule_json = {}
        nodule_json["nodule_id"] = int(nodule_id) # int() float() gets right format for json
        nodule_json["nodule_priority"] = int(patient_annotation["nodule_priority"].loc[patient_annotation["nodule_id"]==nodule_id].iloc[0])
        nodule_json["number_of_annotations"] = len(patient_annotation["nodule_priority"].loc[patient_annotation["nodule_id"]==nodule_id])
        nodule_json["center_yxz_px"] = [int(i) for i in v_center_px]
        nodule_json["center_yxz_mm"] = [float(i) for i in real_center_mm]
        nodule_json["max_diameter_yxz_px"] = [int(i) for i in v_diam_px]
        nodule_json["max_diameter_yxz_mm"] = list(np.array(v_diam_px, dtype=float) * np.array(real_spacing, dtype=float))
        nodule_json["nodule_box_ymin/ymax_xmin/xmax_zmin/zmax_px"] = [int(i) for i in yxz_bbox_px]
        nodule_json["nodule_box_ymin/ymax_xmin/xmax_zmin/zmax_mm"] = [float(i) for i in yxz_bbox_mm]
        nodule_json["nodule_center_box_ymin/ymax_xmin/xmax_zmin/zmax_px"] = [float(i) for i in center_box_coords_yxz_px]
        nodule_json["old_diameter_px"] = int(old_diameter_mm/np.mean(real_spacing[:2]))
        nodule_json["old_diameter_mm"] = float(old_diameter_mm)
        patient_json['nodules'].append(nodule_json)
    saved_path_to_mask = save_nodule_mask(mask_array, folder_name, patient)
    patient_json["mask_path"] = saved_path_to_mask
    nodule_patients_processed_set.add(patient)
    print('processed', len(nodule_patients_processed_set), 'of', len(nodule_patients_set), 'patients', end='\r', flush=True)
    return patient, patient_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pipeline', type=int, required=False, default=False, help="choose num of patients processed")
    args = parser.parse_args()
    if args.test_pipeline:
        test_pipeline = args.test_pipeline
    elif H['test_pipeline']:
        test_pipeline = H['test_pipeline']
    else:
        test_pipeline = False
    dataset_dir = H['dataset_dir']
    dataset_name = H['dataset_name']
    if H['missing_patients'] == True:
        annotations = pd.read_csv(H['LUNA16_missing_annotations_csv_path'], sep=',')
        dataset_json = json.load(open(dataset_dir + dataset_name+'_gen_nodules+masks.json'))
        all_patients = set(annotations['seriesuid'].str.split('.').str[-1])
    else:
        annotations = pd.read_csv(H['LUNA16_annotations_csv_path'], sep=',')
        dataset_json = json.load(open(dataset_dir + dataset_name+'_resample_lungs.json'))
        all_patients = dataset_json['all_patients']
    test_suffix = ''
    folder_name = H["nodule_seg_masks"]
    yx_buffer_px = H['mask_generation_yx_buffer_px']
    z_buffer_px = H['mask_generation_z_buffer_px']
    dataset_json['mask_generation_yx_buffer_px'] = float(yx_buffer_px)
    dataset_json['mask_generation_z_buffer_px'] = float(z_buffer_px)
    ensure_dir(dataset_dir + folder_name + test_suffix + '/')
    # Skip all nodule free patient masks for now.
    # ensure_dir(dataset_dir +'nodules+masks/nodule_free/')
    if test_pipeline:
        all_patients = all_patients[:test_pipeline]
    # load annotations.csv including nodule positions (mm)
    annotations['seriesuid'] = annotations['seriesuid'].str.split('.').str[-1]
    # annotations = annotations.dropna()
    # all patients in list have nodules, not a single patient without nodules.
    nodule_patients_set = set(annotations['seriesuid'].values.tolist())
    if test_pipeline:
        nodule_patients_set = nodule_patients_set & set(all_patients)
    nodule_patients_lst = list(nodule_patients_set)

    # see the next for-loop for a serial version
    n_jobs = min(H['gen_resampled_lung_num_threads'], len(nodule_patients_lst))
    nodule_patients_annotations = dict(Parallel(n_jobs=n_jobs)(delayed(process_nodule_patient)(patient, dataset_json['patients'][patient]) for patient in nodule_patients_lst))
    for patient_cnt, patient in enumerate(tqdm(all_patients)):
        if patient in set(nodule_patients_lst):
            patient_json = dataset_json['patients'][patient]
            # _, dataset_json['patients'][patient] = process_nodule_patient(patient, patient_json) # serial version
            dataset_json['patients'][patient] = nodule_patients_annotations[patient]
        else:
            patient_json = dataset_json['patients'][patient]
            patient_json["nodule_patient"] = False
            img_array = get_img_array(patient_json['resampled_lung_path'])
            mask_array = np.zeros(list(img_array.shape)+[2],dtype=np.uint8)
            saved_path_to_mask = save_nodule_mask(mask_array, folder_name, patient)
            patient_json["mask_path"] = saved_path_to_mask

    filename = dataset_dir + dataset_name + '_gen_nodules+masks' + test_suffix + '.json'
    json.dump(dataset_json, open(filename, 'w'), indent=4)
    print('wrote', filename)

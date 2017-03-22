import os, sys
import numpy as np
import SimpleITK as sitk
from glob import glob
import pandas as pd
import cv2
import scipy.ndimage
from joblib import Parallel, delayed
from collections import OrderedDict
from tqdm import tqdm
from ..utils.ellipse_helpers import *
from .. import params # this shouldn't actually be necessary, but avoids global variables and more complicated stuff
from .. import pipeline as pipe
from .. import visualize as vis

def run(LUNA16_annotations_csv_path,
        ellipse_mode,
        yx_buffer_px,
        z_buffer_px, 
        mask2pred_upper_radius_limit_px,
        # mask2pred_lower_radius_limit_px, # directly read from params.gen_nodule_masks and not checked!
        # reduced_mask_radius_fraction # directly read from params.gen_nodule_masks and not checked!
        **kwargs): # just a hack, as here, we are using the params module
    annotations = pd.read_csv(LUNA16_annotations_csv_path, sep=',')
    # load annotations.csv including nodule positions (mm)
    annotations['seriesuid'] = annotations['seriesuid'].str.split('.').str[-1]
    # all patients in list have nodules, not a single patient without nodules.
    nodule_patients_set = set(annotations['seriesuid'].values.tolist()) & set(pipe.patients)
    # process nodule patients
    resample_lungs_dict = pipe.load_step('resample_lungs')
    pipe.log.info('parallel processing of nodule patients')
    patients_dict = dict(Parallel(n_jobs=min(pipe.n_CPUs, len(nodule_patients_set)), verbose=100)(
                         delayed(process_nodule_patient)(patient, annotations, resample_lungs_dict,
                                                         ellipse_mode, yx_buffer_px, z_buffer_px, mask2pred_upper_radius_limit_px)
                         for patient in nodule_patients_set))
    # loop over non-nodule patients
    pipe.log.info('loop over non_nodule patients')
    for patient in tqdm(pipe.patients):
        if patient not in nodule_patients_set:
            patients_dict[patient] = OrderedDict()
            patients_dict[patient]['nodule_patient'] = False
            img_array = pipe.load_array(resample_lungs_dict[patient]['basename'], 'resample_lungs')
            mask_array = np.zeros(list(img_array.shape) + [2], dtype=np.uint8)
            patients_dict[patient]['basename'] = basename = patient + '_mask.npy'
            patients_dict[patient]['mask_path'] = pipe.save_array(basename, mask_array)
    return patients_dict

def process_nodule_patient(patient, annotations, resample_lungs_dict, 
                           ellipse_mode, yx_buffer_px, z_buffer_px, mask2pred_upper_radius_limit_px): # gen_nodule_masks parameters
    patient_annotation = annotations[annotations['seriesuid'] == patient]
    patient_dict = {}
    patient_dict['nodule_patient'] = True
    patient_dict['number_of_nodules'] = len(set(patient_annotation['nodule_id']))
    patient_dict['nodules'] = []
    resample_lungs_dict_patient = resample_lungs_dict[patient]
    img_array_zyx = pipe.load_array(resample_lungs_dict_patient['basename'], 'resample_lungs')
    bound_box_offset_yx_px = resample_lungs_dict_patient['bound_box_coords_yx_px']
    bound_box_offset_zyx_px = [0, bound_box_offset_yx_px[0], bound_box_offset_yx_px[2]] # no offset in z
    real_spacing_zyx = resample_lungs_dict_patient['resampled_scan_spacing_zyx_mm']
    raw_spacing_zyx = resample_lungs_dict_patient['raw_scan_spacing_zyx_mm']
    origin_zyx = resample_lungs_dict_patient['raw_scan_origin_zyx_mm']
    mask_array_zyx = np.zeros(list(img_array_zyx.shape) + [2], dtype=np.uint8)
    for nodule_id in set(patient_annotation['nodule_id']):
        nodule_annotations = patient_annotation.loc[patient_annotation['nodule_id'] == nodule_id]
        # write nodules to mask_array_zyx
        result = make_nodule(patient, nodule_annotations,
                             mask_array_zyx, img_array_zyx.copy(),
                             origin_zyx, real_spacing_zyx, bound_box_offset_zyx_px, 
                             ellipse_mode, yx_buffer_px, z_buffer_px, mask2pred_upper_radius_limit_px)
        mask_array_zyx, v_center_zyx_px, real_center_mm, v_diam_px, old_diameter_mm, zyx_bbox_px, center_box_coords_zyx_px = result
        zyx_bbox_mm = [(zyx_bbox_px[i] + bound_box_offset_zyx_px[i // 2]) * real_spacing_zyx[i // 2] + origin_zyx[i // 2] for i in range(6)]
        nodule_dict = OrderedDict()
        nodule_dict['nodule_id'] = int(nodule_id) # int() float() gets right format for json
        nodule_dict['nodule_priority'] = int(patient_annotation['nodule_priority'].loc[patient_annotation['nodule_id'] == nodule_id].iloc[0])
        nodule_dict['number_of_annotations'] = len(patient_annotation['nodule_priority'].loc[patient_annotation['nodule_id'] == nodule_id])
        nodule_dict['center_zyx_px'] = [int(i) for i in v_center_zyx_px]
        nodule_dict['center_zyx_mm'] = [float(i) for i in real_center_mm]
        nodule_dict['max_diameter_zyx_px'] = [int(i) for i in v_diam_px]
        nodule_dict['max_diameter_zyx_mm'] = list(np.array(v_diam_px, dtype=float) * np.array(real_spacing_zyx, dtype=float))
        nodule_dict['nodule_box_zmin/zmax_ymin/ymax_xmin/xmax_px'] = [int(i) for i in zyx_bbox_px]
        nodule_dict['nodule_box_zmin/zmax_ymin/ymax_xmin/xmax_mm'] = [float(i) for i in zyx_bbox_mm]
        nodule_dict['nodule_center_box_zmin/zmax_px_ymin/ymax_xmin/xmax'] = [float(i) for i in center_box_coords_zyx_px]
        nodule_dict['old_diameter_px'] = int(old_diameter_mm / np.mean(real_spacing_zyx[:2]))
        nodule_dict['old_diameter_mm'] = float(old_diameter_mm)
        patient_dict['nodules'].append(nodule_dict)
        # Look at the center of the annotation
        for crop in [False, True]:
            if crop:
                cropy = zyx_bbox_px[2], zyx_bbox_px[3]
                cropx = zyx_bbox_px[4], zyx_bbox_px[5]
            else:
                cropy = [0, img_array_zyx.shape[1]]
                cropx = [0, img_array_zyx.shape[2]]
            plt.imshow(img_array_zyx[v_center_zyx_px[0], cropy[0]:cropy[1], cropx[0]:cropx[1]] + 0.25, cmap='Greys')
            color = 'r' if nodule_dict['nodule_priority'] >= 3 else 'orange' if nodule_dict['nodule_priority'] == 2 else 'green'
            level = 255 if nodule_dict['nodule_priority'] >= 3 else 170 if nodule_dict['nodule_priority'] == 2 else 85
            plt.contour(mask_array_zyx[v_center_zyx_px[0], cropy[0]:cropy[1], cropx[0]:cropx[1], 0], colors=color, levels=[level-1], linestyles='dashed')
            plt.contour(mask_array_zyx[v_center_zyx_px[0], cropy[0]:cropy[1], cropx[0]:cropx[1], 1], colors=color, levels=[level-1])
            plt.savefig(pipe.get_step_dir() + 'figs/' + patient + '-nodule' + str(nodule_id) + '_crop' + str(int(crop)) + '.jpg')            
            plt.clf()
    patient_dict['basename'] = basename = patient + '_mask.npy'
    patient_dict['mask_path'] = pipe.save_array(basename, mask_array_zyx)
    return patient, patient_dict

def make_nodule(patient, nodule_annotations,
                mask_array_zyx, img_array_zyx,
                origin_zyx, real_spacing_zyx, bound_box_offset_zyx_px, # resample_lungs parameters
                ellipse_mode, yx_buffer_px, z_buffer_px, mask2pred_upper_radius_limit_px): # gen_nodule_masks parameters
    upper_limit_px = mask2pred_upper_radius_limit_px
    new_mask_array_zyx = np.zeros_like(mask_array_zyx, dtype=np.uint8)
    new_mask_array_zyx_shell = new_mask_array_zyx[:, :, :, 0]
    new_mask_array_zyx_center = new_mask_array_zyx[:, :, :, 1]
    nodule_annotations.sort_values('coordZ')
    # converting coordinates to pixels
    nodule_annotations['coordZ_px'] = ((nodule_annotations['coordZ'].copy() - origin_zyx[0])/real_spacing_zyx[0]
                                       - bound_box_offset_zyx_px[0]).round(decimals=0).astype(int)
    nodule_annotations['coordY_px'] = ((nodule_annotations['coordY'].copy() - origin_zyx[1])/real_spacing_zyx[1] 
                                       - bound_box_offset_zyx_px[1]).round(decimals=0).astype(int)
    nodule_annotations['coordX_px'] = ((nodule_annotations['coordX'].copy() - origin_zyx[2])/real_spacing_zyx[2]
                                        - bound_box_offset_zyx_px[2]).round(decimals=0).astype(int)
    nodule_annotations['diameter_y_px'] = [min(va/real_spacing_zyx[1] + yx_buffer_px, upper_limit_px) for va in nodule_annotations['diameter_y_mm']]
    nodule_annotations['diameter_x_px'] = [min(va/real_spacing_zyx[2] + yx_buffer_px, upper_limit_px) for va in nodule_annotations['diameter_x_mm']]
    center_anno_zyx_mm = [np.mean(nodule_annotations['coordZ']),
                          np.mean(nodule_annotations['coordY']),
                          np.mean(nodule_annotations['coordX'])]
    # no buffer
    z_min_px = int(np.ceil((nodule_annotations['z_min_mm'].iloc[0] - origin_zyx[0])/real_spacing_zyx[0]))
    z_max_px = int(np.floor((nodule_annotations['z_max_mm'].iloc[0] - origin_zyx[0])/real_spacing_zyx[0]))
    # limit size to max 24 px
    z_min_px = max(z_min_px + (z_max_px - z_min_px)//2 - upper_limit_px//2, z_min_px)
    z_max_px = min(z_max_px - (z_max_px - z_min_px)//2 + upper_limit_px//2, z_max_px)
    v_center_zyx_px = [int(round(np.mean(nodule_annotations['coordZ_px']))),
                       int(round(np.mean(nodule_annotations['coordY_px']))),
                       int(round(np.mean(nodule_annotations['coordX_px'])))]
    # diameters at least 2 px
    v_diam_zyx_px = [np.abs(z_max_px-z_min_px),
                     np.max(nodule_annotations['diameter_y_mm']/real_spacing_zyx[1]),
                     np.max(nodule_annotations['diameter_x_mm']/real_spacing_zyx[2])]
    v_diam_zyx_px = [max(2, v) for v in v_diam_zyx_px]
    old_diameter_mm = np.max(nodule_annotations['diameter_mm'])
    nodule_priority = nodule_annotations['nodule_priority'].iloc[0]
    if nodule_priority >= 3:
        nodule_priority_uint8 = 255
    elif nodule_priority == 2:
        nodule_priority_uint8 = 170
    elif nodule_priority == 1:
        nodule_priority_uint8 = 85
    else:
        raise ValueError('Wrong nodule priority in patient ' + patient + '.')
    start_layer = max(0, z_min_px - z_buffer_px)
    end_layer = min(new_mask_array_zyx[:, :, :, 0].shape[0] - 1,
                    (z_max_px + z_buffer_px))
    affected_layers = list(range(start_layer, end_layer + 1))

    # draw full thickness if thickness == -1, otherwise restrict to thickness 2
    small_enough = v_diam_zyx_px[0] * v_diam_zyx_px[1] * v_diam_zyx_px[2] <= 1000    
    thickness = -1 if small_enough or ellipse_mode else 2
    for idx, v_layer in enumerate(affected_layers):
        # in z, y, x
        radius_decrease = 1 # decrease radii factor
        # Case 1: z has its own x_rad and y_rad
        # middle = int(round((z_max_px - z_min_px)/2 + z_min_px))
        if z_min_px <= v_layer <= z_max_px:
            # print('middle')
            # position of the v_layer in mm
            CoordZ_mm = (v_layer + bound_box_offset_zyx_px[0]) * real_spacing_zyx[0] + origin_zyx[0]
            # taking the annotation that is closest to the layer´s z-position
            x_rad = nodule_annotations['diameter_x_px'].iloc[(nodule_annotations['coordZ'] - CoordZ_mm).abs().argsort().iloc[0]] // 2
            y_rad = nodule_annotations['diameter_y_px'].iloc[(nodule_annotations['coordZ'] - CoordZ_mm).abs().argsort().iloc[0]] // 2
            x_center = nodule_annotations['coordX_px'].iloc[(nodule_annotations['coordZ'] - CoordZ_mm).abs().argsort().iloc[0]]
            y_center = nodule_annotations['coordY_px'].iloc[(nodule_annotations['coordZ'] - CoordZ_mm).abs().argsort().iloc[0]]
            radii_yx = (int(y_rad), int(x_rad)) # TODO: check
        # Case 2: z does not have its x_rad and y_rad, due to buffering
        else:
            if v_layer < z_min_px:
                # print('below')
                x_rad = nodule_annotations['diameter_x_px'].ix[nodule_annotations['coordZ'].idxmin()] // 2
                y_rad = nodule_annotations['diameter_y_px'].ix[nodule_annotations['coordZ'].idxmin()] // 2
                x_center = nodule_annotations['coordX_px'].ix[nodule_annotations['coordZ'].idxmin()]
                y_center = nodule_annotations['coordY_px'].ix[nodule_annotations['coordZ'].idxmin()]
                radii_yx = (int(round(np.sqrt(max(1, y_rad**2 - (radius_decrease*(z_min_px - v_layer))**2)))), # TODO: check!
                                   int(round(np.sqrt(max(1, x_rad**2 - (radius_decrease*(z_min_px - v_layer))**2)))))
            elif v_layer > z_max_px:
                # print('above')
                x_rad = nodule_annotations['diameter_x_px'].ix[nodule_annotations['coordZ'].idxmax()] // 2
                y_rad = nodule_annotations['diameter_y_px'].ix[nodule_annotations['coordZ'].idxmax()] // 2
                x_center = nodule_annotations['coordX_px'].ix[nodule_annotations['coordZ'].idxmax()]
                y_center = nodule_annotations['coordY_px'].ix[nodule_annotations['coordZ'].idxmax()]
                radii_yx = (int(round(np.sqrt(max(1, y_rad**2 - (radius_decrease * (z_max_px - v_layer))**2)))), # TODO: check
                            int(round(np.sqrt(max(1, x_rad**2 - (radius_decrease * (z_max_px - v_layer))**2)))))
            else:
                log.pipe.warning('untreated layer in patient', patient)
        radii_yx = tuple([max(2, v) for v in radii_yx])
        # get an array for drawing with cv2
        draw_array = new_mask_array_zyx_shell[v_layer, :, :].copy()
        # for calling cv2.ellipse, need to change the convention to xy for center and axes but keep the array the same (i.e. yx)
        center_draw_xy = (x_center, y_center)
        radii_draw_xy = (int(radii_yx[1]), int(radii_yx[0]))
        cv2.ellipse(draw_array, center=center_draw_xy, axes=radii_draw_xy,
                    angle=0, startAngle=0, endAngle=360, color=(nodule_priority_uint8), thickness=thickness)
        if draw_array.shape[0] > draw_array.shape[1]:
            draw_array = np.swapaxes(draw_array, 0, 1)
        new_mask_array_zyx_shell[v_layer, :, :] = np.clip(draw_array + new_mask_array_zyx_shell[v_layer, :, :], 0, 255)

    if ellipse_mode:
        result = draw_ellipse(new_mask_array_zyx_shell, nodule_priority_uint8, v_center_zyx_px, v_diam_zyx_px)
        new_mask_array_zyx_shell, bbox_px_shell_zyx, new_mask_array_zyx_center, bbox_px_center_zyx = result
    else:
        bbox_px_shell_zyx = get_bounding_box(new_mask_array_zyx_shell)
        bbox_px_center_zyx = get_bounding_box(new_mask_array_zyx_shell) # If we do not fit ellipsoids, take the shell annotation also for the center.

    # Ensure nodule priority in case of overlap - always take maximum
    mask_array_zyx[:, :, :, 0] = np.maximum(new_mask_array_zyx_shell, mask_array_zyx[:, :, :, 0])
    mask_array_zyx[:, :, :, 1] = np.maximum(new_mask_array_zyx_center, mask_array_zyx[:, :, :, 1])            
    if 0: # draw center in center channel
        randy = np.random.randint(0, 1000)
        for cnt, z_layer in enumerate(np.arange(v_center_zyx_px[2] - 2, v_center_zyx_px[2] + 2)):
            cv2.imwrite('test_imgs/' + patient+ '_' + str(randy) + '_center_' + str(cnt)+'.jpg', mask_array_zyx[:,:,z_layer,1])
            cv2.imwrite('test_imgs/' + patient+ '_' + str(randy) + '_img_' + str(cnt)+'.jpg', (255/1400.*img_array_zyx[z_layer, :, :]).astype(np.uint8))
            cv2.imwrite('test_imgs/' + patient+ '_' + str(randy) + '_shell_' + str(cnt)+'.jpg', mask_array_zyx[:,:,z_layer,0])
    return mask_array_zyx, v_center_zyx_px, center_anno_zyx_mm, v_diam_zyx_px, old_diameter_mm, bbox_px_shell_zyx, bbox_px_center_zyx

def draw_ellipse(mask_array_zyx, color, v_center_px, v_diam_px):
    v_diam_px = 2 * v_diam_px
    X = np.argwhere(mask_array_zyx > 1)
    center, radii, rotation = getMinVolEllipse(X, tolerance=0.01, v_center_px=v_center_px, v_diam_px=v_diam_px)
    radii_shell = [max(params.gen_nodule_masks['mask2pred_lower_radius_limit_px'], int(r)) for r in radii]
    new_mask_shell, bbox_px_shell = draw_new_ellipsoid(mask_array_zyx.shape, 
                                                       center,
                                                       radii_shell,
                                                       rotation, v_center_px, v_diam_px, color)
    # draw second mask with reduced size
    radii_center = [max(1, int(r*float(params.gen_nodule_masks['reduced_mask_radius_fraction']))) for r in radii]
    new_mask_center, bbox_px_center = draw_new_ellipsoid(mask_array_zyx.shape,
                                                         center,
                                                         radii_center,
                                                         rotation, v_center_px, v_diam_px, color)
    # print(np.unique(new_mask, return_counts=True))
    # plot__both_scatters(new_points, X)
    return new_mask_shell, bbox_px_shell, new_mask_center, bbox_px_center

def draw_new_ellipsoid(new_mask_shape, center, radii, rotation, v_center_px, v_diam_px, color):
    new_mask = np.zeros(new_mask_shape, dtype=np.uint8)
    for i in range(max(0, v_center_px[0] - int(round(v_diam_px[0])) - 15), min(v_center_px[0] + int(round(v_diam_px[0])) + 15, new_mask.shape[0] - 1)):
        for j in range(max(0, v_center_px[1] - int(round(v_diam_px[1])) - 15), min(v_center_px[1] + int(round(v_diam_px[1])) + 15, new_mask.shape[1] - 1)):
            for k in range(max(0, v_center_px[2] - int(round(v_diam_px[2])) - 15), min(v_center_px[2] + int(round(v_diam_px[2])) + 15, new_mask.shape[2] - 1)):
                r = np.array([i, j, k])
                r = np.dot(rotation, r - center)
                c_value = (r[0]/radii[0])**2 + (r[1]/radii[1])**2 + (r[2]/radii[2])**2 # Ellipse constraint
                if c_value <=1:
                    new_mask[i, j, k] = color
    bbox = get_bounding_box(new_mask)
    return new_mask, bbox

def get_bounding_box(array):
    points = np.argwhere(array > 1)
    bbox = [np.min(points[:, 0]), np.max(points[:, 0]),
            np.min(points[:, 1]), np.max(points[:, 1]),
            np.min(points[:, 2]), np.max(points[:, 2])]
    return bbox

def plot(mask, image):
    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()

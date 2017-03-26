import os, sys
import numpy as np
import json
from sklearn.cluster import DBSCAN
from functools import reduce
from tqdm import tqdm
from collections import OrderedDict
from joblib import Parallel, delayed
from .. import pipeline as pipe
from .. import utils

# rank the candidatess / clusters according to the following score
sort_clusters_by = 'prob_sum_min_nodule_size'

def run(n_candidates,
        threshold_prob_map,
        cube_shape,
        all_patients):
    resample_lungs_json = pipe.load_json('out.json', 'resample_lungs')
    gen_prob_maps_json = pipe.load_json('out.json', 'gen_prob_maps')
    gen_nodule_masks_json = None
    considered_patients = pipe.patients if all_patients else pipe.patients_by_split['va']
    if pipe.dataset_name == 'LUNA16':
        gen_nodule_masks_json = pipe.load_json('out.json', 'gen_nodule_masks')
    patients_candidates_json = OrderedDict(Parallel(n_jobs=min(pipe.n_CPUs, len(considered_patients)), verbose=100)(
                                           delayed(process_patient)(patient,
                                                                    n_candidates,
                                                                    threshold_prob_map,
                                                                    cube_shape,
                                                                    resample_lungs_json, 
                                                                    gen_prob_maps_json,
                                                                    gen_nodule_masks_json)
                                           for patient in considered_patients))
    # write both patients and candidates list
    patients_lst_path = pipe.get_step_dir() + 'patients.lst'
    patients_lst = open(patients_lst_path, 'w')
    # if pipe.dataset_name == 'LUNA16':
    if 1:
        candidates_lst_path = pipe.get_step_dir() + 'candidates.lst'
        candidates_lst = open(candidates_lst_path, 'w')
    for patient_cnt, patient in enumerate(tqdm(considered_patients)):
        patients_lst.write(patients_candidates_json[patient]['patients_lst'])
        del patients_candidates_json[patient]['patients_lst']
        # if pipe.dataset_name == 'LUNA16':
        if 1:
            for line in patients_candidates_json[patient]['candidates_lst']:
                candidates_lst.write(line)
            del patients_candidates_json[patient]['candidates_lst']
    patients_lst.close()
    print('wrote', patients_lst_path)
    # if pipe.dataset_name == 'LUNA16':
    if 1:
        candidates_lst.close()
        print('wrote', candidates_lst_path)
    pipe.save_json('out.json', patients_candidates_json)

def process_patient(patient,
                    n_candidates,
                    threshold_prob_map,
                    cube_shape,
                    resample_lungs_json,
                    gen_prob_maps_json,
                    gen_nodule_masks_json):
    prob_map = pipe.load_array(gen_prob_maps_json[patient]['basename'], 'gen_prob_maps')
    if prob_map.dtype == np.float32:
        prob_map = (255 * prob_map).astype(np.uint8)
    elif prob_map.dtype == np.uint16:
        raise ValueError('Data type unit16 for prob_map not implemented in gen_candidates.')
    prob_map_thresh = prob_map.copy()
    prob_map_thresh[prob_map_thresh < threshold_prob_map * 255] = 0.0 # here prob_map is in units of 255
    # the points in mm units, relative to dummy origin in pixels
    prob_map_points_px = np.argwhere(prob_map_thresh)
    prob_map_points_mm = prob_map_points_px * resample_lungs_json[patient]['resampled_scan_spacing_zyx_mm']
    try:
        avg_n_points_per_cmm = int(np.round(reduce(lambda x, y: x*y, 
                                                   [1/s for s in resample_lungs_json[patient]['resampled_scan_spacing_zyx_mm']])))
    except:
        avg_n_points_per_cmm = 4
        wrong_spacing_warning = ('Wrong resampled spacing data {} for patient {}.'.format(
                                 resample_lungs_json[patient]['resampled_scan_spacing_zyx_mm'], patient))
        pipe.log.error(wrong_spacing_warning + ' Assuming per cmm ' + str(avg_n_points_per_cmm))

    prob_map_X_norm = prob_map[prob_map_points_px[:, 0], prob_map_points_px[:, 1], prob_map_points_px[:, 2]].astype('float32') / 255
    dbscan_args = prob_map_points_mm, prob_map_points_px, prob_map_X_norm, avg_n_points_per_cmm
    clusters = dbscan(*dbscan_args)
    # the clusters might be overly large, split them if this is the case
    clusters = split_clusters(clusters, cube_shape, prob_map.shape, dbscan_args,
                              threshold=threshold_prob_map)
    # now we are sure that the clusters are small enough to fit into the boxes
    # get the boxes around the clusters and the corresponding arrays
    clusters = get_clusters_box_coords(clusters, cube_shape)
    clusters = get_clusters_array(clusters, cube_shape, prob_map, 'prob_map')
    # sort clusters, trunkate clusters, clean the cluster dict
    clusters = sort_clusters(clusters, key=sort_clusters_by)
    clusters = clusters[:n_candidates]
    clusters = remove_masks_from_clusters(clusters)
    # fill clusters with the original image data
    img_array = pipe.load_array(resample_lungs_json[patient]['basename'], 'resample_lungs')
    clusters = get_clusters_array(clusters, cube_shape, img_array.copy(), 'img')
    patient_json = OrderedDict()
    if pipe.dataset_name == 'dsb3':
        # set cancer labels
        patient_json['label'] = pipe.patients_label[patient]
    else:
        # set nodule labels for candidates
        gen_nodule_masks_json_patient = gen_nodule_masks_json[patient]
        count_nodules_prio_greater_2 = 0
        for clu in clusters:
            clu['nodule_priority'] = 0 # init each cluster with prio 0
            if gen_nodule_masks_json_patient['nodule_patient']:
                nodules = gen_nodule_masks_json_patient['nodules']
                for nodule_cnt, nodule in enumerate(nodules):
                    nodule_center = nodule['center_zyx_px']
                    can_center = clu['center_px']
                    if is_contained(can_center, nodule_center, cube_shape):
                        clu['nodule_priority'] = max(nodule['nodule_priority'], clu['nodule_priority'])
                        if clu['nodule_priority'] > 2:
                            count_nodules_prio_greater_2 += 1
                            break
        # generate a list of the non-detected nodules
        non_detected_nodules = []
        if gen_nodule_masks_json_patient['nodule_patient']:
            nodules = gen_nodule_masks_json_patient['nodules']
            for nodule_cnt, nodule in enumerate(nodules):
                nodule_center = nodule['center_zyx_px']
                nodule_is_contained = False
                for clu in clusters:
                    can_center = clu['center_px']
                    if is_contained(can_center, nodule_center, cube_shape):
                        nodule_is_contained = True
                if not nodule_is_contained:
                    non_detected_nodules.append(nodule_cnt)                
        patient_json['label'] = count_nodules_prio_greater_2 > 0
    patient_json['clusters'] = [] # this stores all the information about the candidates
    patient_json['candidates_lst'] = [] # this is a sorted list of candidates
    can_img_paths = []; can_prob_map_paths = []
    for cluster_cnt, clu in enumerate(clusters):
        patient_json['clusters'].append(OrderedDict())
        cluster_json = patient_json['clusters'][cluster_cnt]
        # save candidate from img_array
        cluster_json['img_basename'] = basename_img = patient + '_{:02}_img.npy'.format(cluster_cnt)
        cluster_json['img_path'] = pipe.save_array(basename_img, clu['img_array'].astype(np.float32))
        can_img_paths.append(cluster_json['img_path'])
        # save candidate from prob_map
        cluster_json['prob_map_basename'] = basename_prob_map = patient + '_{:02}_prob_map.npy'.format(cluster_cnt)
        cluster_json['prob_map_path'] = pipe.save_array(basename_prob_map, clu['prob_map_array'].astype(np.uint8))
        can_prob_map_paths.append(cluster_json['prob_map_path'])
        # check cluster_shape
        if clu['prob_map_array'].shape != tuple(cube_shape):
            raise ValueError('Wrong shape {} for patient  {}'.format(clu['prob_map_array'].shape, patient))
        # save some cluster info
        cluster_json['is_non_detect'] = 0
        cluster_json['prob_max_cluster'] = int(clu['prob_max_cluster'])  # is all in units of 255
        cluster_json['prob_sum_cluster'] = int(clu['prob_sum_cluster'])
        cluster_json['prob_sum_min_nodule_size'] = int(clu['prob_sum_min_nodule_size'])
        cluster_json['size_points_cluster'] = int(clu['size_points_cluster'])
        cluster_json['center_px'] = [int(x) for x in clu['center_px']]
        cluster_json['box_coords_px'] = clu['box_coords_px']
        # candidate classification info
        if pipe.dataset_name == 'LUNA16':
            cluster_json['nodule_priority'] = nodule_priority = clu['nodule_priority']
            is_non_detect = 0
            # account for non-detects only in list
            if cluster_cnt >= len(clusters) - len(non_detected_nodules):
                non_detect_cnt = non_detected_nodules[len(clusters) - cluster_cnt - 1]
                is_non_detect = 1
                cluster_json['is_non_detect'] = 1
                # overwrite the cluster info
                cluster_json['prob_max_cluster'] = 0
                cluster_json['prob_sum_cluster'] = 0
                cluster_json['prob_sum_min_nodule_size'] = 0
                cluster_json['size_points_cluster'] = 0
                cluster_json['center_px'] = [0, 0, 0]
                cluster_json['box_coords_px'] = [0, 0, 0, 0, 0, 0]
                cluster_json['nodule_priority'] = 0
                nodule_priority = nodules[non_detect_cnt]['nodule_priority']
                # overwrite the former candidate
                nodule_box = get_cluster_box_coords(nodules[non_detect_cnt]['center_zyx_px'], cube_shape)
                img_array_non_detect = utils.crop_and_embed(img_array, nodule_box, cube_shape)
                pipe.save_array(basename_img, img_array.astype(np.float32))
                prob_map_array_non_detect = utils.crop_and_embed(prob_map, nodule_box, cube_shape)
                pipe.save_array(basename_prob_map, prob_map_array_non_detect.astype(np.uint8))
            patient_json['candidates_lst'].append('{}_{}\t{}\t{}\t{}\t{}\n'.format(patient, cluster_cnt,
                                                                                   nodule_priority,
                                                                                   cluster_json['img_path'],
                                                                                   cluster_json['prob_map_path'],
                                                                                   is_non_detect))
        elif pipe.dataset_name == 'dsb3':
            patient_json['candidates_lst'].append('{}_{}\t{}\t{}\t{}\n'.format(patient, cluster_cnt, 
                                                                               patient_json['label'], cluster_json['img_path'], cluster_json['prob_map_path']))

    patient_json['patients_lst'] = '{}\t{}\t{}\t{}\n'.format(patient, patient_json['label'],
                                                             ','.join(can_img_paths), ','.join(can_prob_map_paths))
    
    return patient, patient_json

def get_clusters_box_coords(clusters, cube_shape):
    for cluster in clusters:
        cluster['box_coords_px'] = get_cluster_box_coords(cluster['center_px'], cube_shape)
    return clusters

def get_cluster_box_coords(cluster_center, cube_shape):
    box_coords = []
    for coord in range(3):
        start = int(cluster_center[coord] - cube_shape[coord]/2)
        box_coords += [start]
        box_coords += [start + cube_shape[coord]]
    return box_coords

def get_clusters_array(clusters, cube_shape, array, array_name):
    for cluster in clusters:
        cluster[array_name + '_array'] = utils.crop_and_embed(array, cluster['box_coords_px'], cube_shape)
        if array_name == 'img':
            cluster[array_name + '_array'] = cluster[array_name + '_array'].astype(np.int16)
        elif 'prob_map' in array_name:
            cluster[array_name + '_array'] = cluster[array_name + '_array'].astype(np.uint8)
    return clusters

def dbscan(X_mm, X_px, weights, avg_n_points_per_cmm=1):
    """
    Clusters data matrix X_mm.

    avg_n_points_per_cmm : int, optional (default: 1)
       For a grid with spacing 1mm x 1mm x 1mm, this is 1.
       For a grid with spacing .5mm x .5mm x .5mm, this is 8.

    min_nodule_weight_factor : float, optional (default: 1)
       Factor to multiply with avg_n_points_per_cmm to obtain
           min_nodule_weight = min_nodule_weight_factor * avg_n_points_per_cmm 
       which is the lower weight threshold for starting a new cluster.
    """
    # if there are no non-zero entries
    if X_mm.shape[0] == 0: return []
    min_nodule_weight = 0.2 * avg_n_points_per_cmm # this is hard-coded and affects the clustering
    min_nodule_size = int(20 * avg_n_points_per_cmm) # this is hard coded and only affects the ranking
    # epsilon is in units mm, min_samples includes the point itself
    # on 1mm x 1mm x 1mm, we've seen prob_maps with just 4 high prob values that correspond to a nodule
    # only returns core_samples, therefore clusters might be smaller than min_nodule_size
    db = DBSCAN(eps=1.03, min_samples=min_nodule_weight).fit(X_mm, sample_weight=weights)
    if not bool(db): return []
    labels = db.labels_
    # print('unique labels', np.unique(labels))
    loop_over_labels = (label for label in np.unique(labels) if label >= 0)
    clusters = []
    for label in loop_over_labels:
        mask = label == labels
        clu = {}
        clu['mask'] = mask
        clu['center_mm'] = [np.average(X_mm[mask, i].astype(np.float32), weights=weights[mask]).astype(np.int16)
                            for i in range(X_mm.shape[1])]
        clu['center_px'] = [np.average(X_px[mask, i].astype(np.float32), weights=weights[mask]).astype(np.int16)
                            for i in range(X_px.shape[1])]
        clu['max_px'] = [np.max(X_px[mask, i].astype(np.float32)).astype(np.int16)
                         for i in range(X_px.shape[1])]
        clu['min_px'] = [np.min(X_px[mask, i].astype(np.float32)).astype(np.int16)
                         for i in range(X_px.shape[1])]
        clu['array'] = X_px[mask]
        clu['size_points_cluster'] = clu['array'].shape[0]
        clu['prob_max_cluster'] = np.max(weights[mask])
        clu['prob_sum_cluster'] = np.sum(weights[mask])
        # the sum over the highest scoring points up to a number that approximately corresponds to the minimal nodule size
        clu['prob_sum_min_nodule_size'] = np.sum(np.partition(weights[mask], -min_nodule_size)[-min_nodule_size:]
                                                 if clu['size_points_cluster'] > min_nodule_size else weights[mask])
        clusters.append(clu)
    return clusters

def split_clusters(clusters, cube_shape, total_shape, dbscan_args=None, threshold=0.05, check=False):
    clusters_remove_indices = []
    clusters_append = []
    threshold_base = threshold
    threshold_param = 1
    for cluster_cnt, clu in enumerate(clusters):
        cluster_center = clu['center_px']
        can_coords = {}
        for coord, coord_name in enumerate(['y', 'x', 'z']):
            min_ = int(cluster_center[coord] - cube_shape[coord]/2)
            max_ = min_ + cube_shape[coord]
            if clu['min_px'][coord] < min_ or clu['max_px'][coord] > max_:
                # print('cluster', cluster_cnt, 'with center', clu['center_px'], 'too large in', coord_name, 'direction')
                if check:
                    return False
                mask_cluster = clu['mask']
                while threshold_param < 1000:
                    threshold_param *= 1.03
                    threshold = threshold_base + np.tanh(threshold_param-1)*(1-threshold_base)
                    X_mm, X_px, weights, avg_n_points_per_cmm = dbscan_args
                    mask_threshold = weights[mask_cluster] > threshold
                    clusters_split = dbscan(X_mm[mask_cluster][mask_threshold], 
                                            X_px[mask_cluster][mask_threshold], 
                                            weights[mask_cluster][mask_threshold],
                                            avg_n_points_per_cmm=avg_n_points_per_cmm)
                    if split_clusters(clusters_split, cube_shape, total_shape, check=True):
                        if len(clusters_split) > 0:
                            # print('cluster', cluster_cnt, 'split into', len(clusters_split), 'new clusters')
                            # print('add to center', clu['center_px'], 'new', ' '.join([str(c['center_px']) for c in clusters_split]))
                            # print('      weights', clu['prob_sum_cluster'], 'new', ' '.join([str(c['prob_sum_cluster']) for c in clusters_split]))
                            clusters_split = remove_redundant_clusters(clusters_split, cube_shape)
                            clusters_split = remove_redundant_clusters_compare_with_one(clusters_split, clu, cube_shape)
                            # print('       pruned', clu['center_px'], 'new', ' '.join([str(c['center_px']) for c in clusters_split]))
                            # print('             ', clu['prob_sum_cluster'], 'new', ' '.join([str(c['prob_sum_cluster']) for c in clusters_split]))
                            clusters_append += clusters_split
                        else:
                            # it's not so drastical as we rank clusters according to prob_sum_min_nodule_size
                            # print('cluster', cluster_cnt, 'with center', clu['center_px'], 'and weight', clu['prob_sum_cluster'], 'could not be split')
                            clu['not_split'] = True
                        break
                break
    if check:
        return True
    else:
        for i in sorted(clusters_remove_indices, reverse=True):
            del clusters[i]
        clusters += clusters_append
        return clusters

def sort_clusters(clusters, key='prob_sum_cluster'):
    sorted_clusters = sorted(clusters, key=lambda cluster: cluster[key], reverse=True)
    return sorted_clusters

def remove_masks_from_clusters(clusters):
    for cluster in clusters:
        del cluster['mask']
    return clusters

def remove_redundant_clusters(clusters, cube_shape):
    clusters_remove_indices = []
    for iclu, clui in enumerate(clusters):
        for jclu, cluj in enumerate(clusters[:iclu]):
            max_dist_fraction = 0.15
            if is_contained(clui['center_px'], cluj['center_px'], cube_shape, max_dist_fraction):
                if clui['prob_sum_cluster'] > cluj['prob_sum_cluster']:
                    clusters_remove_indices.append(jclu)
                else:
                    clusters_remove_indices.append(iclu)
    for i in sorted(set(clusters_remove_indices), reverse=True):
        del clusters[i]
    return clusters

def remove_redundant_clusters_compare_with_one(clusters, clu, cube_shape):
    clusters_remove_indices = []
    for iclu, clui in enumerate(clusters):
        max_dist_fraction = 0.15
        if is_contained(clui['center_px'], clu['center_px'], cube_shape, max_dist_fraction):
            clusters_remove_indices.append(iclu)
    for i in sorted(set(clusters_remove_indices), reverse=True):
        del clusters[i]
    return clusters

def is_contained(center_1, center_2, cube_shape, max_dist_fraction=0.5):
    return np.all([(abs(c-n) < max_dist_fraction*s) for c, n, s in zip(center_1, center_2, cube_shape)])

import os, sys
import numpy as np
import json
from sklearn.cluster import DBSCAN
from functools import reduce
from tqdm import tqdm
from collections import OrderedDict
from joblib import Parallel, delayed
from .. import pipeline as pipe

# rank the candidatess / clusters according to the following score
sort_clusters_by = 'prob_sum_min_nodule_size'

def run(n_candidates,
        threshold_prob_map,
        cube_shape):
    resample_lungs_json = pipe.load_json('out.json', 'resample_lungs')
    gen_prob_maps_json = pipe.load_json('out.json', 'gen_prob_maps')
    gen_nodule_masks_json = None
    if pipe.dataset_name == 'LUNA16':
        gen_nodule_masks_json = pipe.load_json('out.json', 'gen_nodule_masks')
    patients_candidates_json = OrderedDict(Parallel(n_jobs=min(pipe.n_CPUs, len(pipe.patients)), verbose=100)(
                                           delayed(process_patient)(patient,
                                                                    n_candidates,
                                                                    threshold_prob_map,
                                                                    cube_shape,
                                                                    resample_lungs_json, 
                                                                    gen_prob_maps_json,
                                                                    gen_nodule_masks_json)
                                           for patient in pipe.patients))
    # write both patients and candidates list
    patients_lst_path = pipe.get_step_dir() + 'patients.lst'
    patients_lst = open(patients_lst_path, 'w')
    if pipe.dataset_name == 'LUNA16':
        candidates_lst_path = pipe.get_step_dir() + 'candidates.lst'
        candidates_lst = open(candidates_lst_path, 'w')
    for patient_cnt, patient in enumerate(tqdm(pipe.patients)):
        patients_lst.write(patients_candidates_json[patient]['patients_lst'])
        del patients_candidates_json[patient]['patients_lst']
        if pipe.dataset_name == 'LUNA16':
            for line in patients_candidates_json[patient]['candidates_lst']:
                candidates_lst.write(line)
            del patients_candidates_json[patient]['candidates_lst']
    patients_lst.close()
    print('wrote', patients_lst_path)
    if pipe.dataset_name == 'LUNA16':
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
    clusters = split_clusters(clusters, cube_shape, prob_map.shape, dbscan_args,
                              threshold=threshold_prob_map)
    # print('found', len(clusters), 'clusters in', len(prob_map_points_mm), 'points')
    clusters = get_candidates_box_coords(clusters, cube_shape, prob_map.shape)
    clusters = get_candidates_array(clusters, prob_map, 'prob_map', cube_shape, threshold_prob_map)
    # sort clusters, trunkate clusters, clean the cluster dict
    clusters = sort_clusters(clusters, key=sort_clusters_by)
    clusters = clusters[:n_candidates]
    clusters = remove_masks_from_clusters(clusters)

    img_array = pipe.load_array(resample_lungs_json[patient]['basename'], 'resample_lungs')
    clusters = get_candidates_array(clusters, img_array.copy(), 'img', cube_shape)
    patient_json = OrderedDict()
    if pipe.dataset_name == 'dsb3':
        # set cancer labels
        patient_json['label'] = pipe.patients_labels[patient]
    else:
        # set nodule labels
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
        patient_json['label'] = count_nodules_prio_greater_2 > 0
    patient_json['clusters'] = [] # this is a sorted list of candidates
    patient_json['candidates_lst'] = []
    can_img_paths = []; can_prob_map_paths = []
    for cluster_cnt, clu in enumerate(clusters):
        patient_json['clusters'].append({})
        cluster_json = patient_json['clusters'][cluster_cnt]
        # save candidate from img_array
        cluster_json['img_basename'] = basename = patient + '_%02d_img.npy' % (cluster_cnt)
        cluster_json['img_path'] = pipe.save_array(basename, clu['candidate_img_array'].astype(np.float32))
        can_img_paths.append(cluster_json['img_path'])
        # save candidate from prob_map
        basename = patient + '_%02d_prob_map.npy'
        cluster_json['prob_map_basename'] = basename
        cluster_json['prob_map_path'] = pipe.save_array(basename, clu['candidate_prob_map_array'].astype(np.float32))
        can_prob_map_paths.append(cluster_json['prob_map_path'])
        # check cluster_shape
        if clu['candidate_prob_map_array'].shape != tuple(cube_shape):
            pipe.log.error('Wrong shape {} for patient  {}'.format(clu['candidate_prob_map_array'].shape, patient))
        # save some cluster info
        cluster_json['prob_max_cluster'] = int(clu['prob_max_cluster'])  # is all in units of 255
        cluster_json['prob_sum_cluster'] = int(clu['prob_sum_cluster'])
        cluster_json['prob_sum_candidate'] = int(clu['prob_sum_candidate'])
        cluster_json['prob_sum_min_nodule_size'] = int(clu['prob_sum_min_nodule_size'])
        cluster_json['size_points_cluster'] = int(clu['size_points_cluster'])
        cluster_json['center_px'] = [int(x) for x in clu['center_px']]
        cluster_json['candidate_box_coords_zyx_px'] = [int(clu['candidate_box_coords'][k]) for k  in ['min_z', 'max_z', 'min_y', 'max_y', 'min_x', 'max_x']]
        # candidate classification info
        if pipe.dataset_name == 'LUNA16':
            patient_json['candidates_lst'].append('{}_{}\t{}\t{}\t{}\n'.format(patient, cluster_cnt, 
                                                                               clu['nodule_priority'], cluster_json['img_path'], cluster_json['prob_map_path']))
    patient_json['patients_lst'] = '{}\t{}\t{}\t{}\n'.format(patient, patient_json['label'],
                                                             ','.join(can_img_paths), ','.join(can_prob_map_paths))
    return patient, patient_json

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
    min_nodule_weight = 0.2 * avg_n_points_per_cmm
    min_nodule_size = int(20 * avg_n_points_per_cmm)
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

def split_clusters(clusters, cube_shape, total_shape, dbscan_args, threshold=0.05, check=False, padding=True):
    clusters_remove_indices = []
    clusters_append = []
    threshold_base = threshold
    threshold_param = 1
    for cluster_cnt, clu in enumerate(clusters):
        cluster_center = clu['center_px']
        can_coords = {}
        for coord, coord_name in enumerate(['y', 'x', 'z']):
            min_ = int(cluster_center[coord]-cube_shape[coord]/2)
            max_ = min_ + cube_shape[coord]
            if clu['min_px'][coord] < min_ or clu['max_px'][coord] > max_:
                # print('cluster', cluster_cnt, 'with center', clu['center_px'], 'too large in', coord_name, 'direction')
                if check:
                    return False
                mask_cluster = clu['mask']
                while threshold_param < 1000:
                    threshold_param *= 1.03
                    threshold = threshold_base + np.tanh(threshold_param-1)*(1-threshold_base)
                    # print(threshold, end=' ')
                    mask_threshold = weights[mask_cluster] > threshold
                    X_mm, X_px, weights, avg_n_points_per_cmm = dbscan_args
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

def get_candidates_box_coords(clusters, cube_shape, total_shape, padding=True):
    for cluster in clusters:
        cluster_center = cluster['center_px']
        can_coords = {}
        for coord, coord_name in enumerate(['y', 'x', 'z']):
            if padding:
                min_ = int(cluster_center[coord]-cube_shape[coord]/2)
                max_ = min_ + cube_shape[coord]
            else:
                min_ = int(min(max(cluster_center[coord]-cube_shape[coord]/2, 0), total_shape[coord]-cube_shape[coord]))
                max_ = min_ + cube_shape[coord]
            can_coords['min_' + coord_name] = min_
            can_coords['max_' + coord_name] = max_
        cluster['candidate_box_coords'] = can_coords
    return clusters

def get_candidates_array(clusters, array, array_name, cube_shape, threshold_prob_map=0.05):
    total_shape = array.shape
    for cluster_cnt, cluster in enumerate(clusters):
        coords = cluster['candidate_box_coords']
        if array_name == 'img' and array.dtype != np.int16:
            raise ValueError('Wrong data type.')
        if array_name == 'prob_map' and array.dtype != np.uint8:
            raise ValueError('Wrong data type.')
        if array_name=='img':
            cluster['candidate_'+array_name+'_array'] = np.zeros(cube_shape, dtype=np.int16)
        elif array_name=='prob_map':
            cluster['candidate_'+array_name+'_array'] = np.zeros(cube_shape, dtype=np.uint8)

        cluster['candidate_'+array_name+'_array'][abs(coords['min_y']) if coords['min_y']<0 else 0: cube_shape[0] if coords['max_y']<=total_shape[0] else -(coords['max_y']-total_shape[0]),
                                              abs(coords['min_x']) if coords['min_x']<0 else 0: cube_shape[1] if coords['max_x']<=total_shape[1] else -(coords['max_x']-total_shape[1]),
                                              abs(coords['min_z']) if coords['min_z']<0 else 0: cube_shape[2] if coords['max_z']<=total_shape[2] else -(coords['max_z']-total_shape[2])]\
                                                      = array[0 if coords['min_y']<0 else coords['min_y']: total_shape[0] if coords['max_y']>total_shape[0] else coords['max_y'],
                                                              0 if coords['min_x']<0 else coords['min_x']: total_shape[1] if coords['max_x']>total_shape[1] else coords['max_x'],
                                                              0 if coords['min_z']<0 else coords['min_z']: total_shape[2] if coords['max_z']>total_shape[2] else coords['max_z']]
        if array_name == 'prob_map':
            mask_threshold = cluster['candidate_prob_map_array'] > threshold_prob_map * 255
            cluster['prob_sum_candidate'] = np.sum(cluster['candidate_prob_map_array'][mask_threshold])
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

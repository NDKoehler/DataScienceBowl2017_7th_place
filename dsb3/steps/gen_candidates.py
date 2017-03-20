import numpy as np
import pandas as pd
import os, sys
from skimage import measure
from sklearn.cluster import DBSCAN
from functools import reduce
import json
from natsort import natsorted
import argparse
import time
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
from master_config import H
from utils import ensure_dir

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

def split_clusters(clusters, cube_shape, total_shape,
                   X_mm=None, X_px=None, weights=None, avg_n_points_per_cmm=1, threshold=0.05, check=False, padding=True):
    clusters_remove_indices = []
    clusters_append = []
    threshold_base = threshold
    threshold_param = 1
    for clu_cnt, clu in enumerate(clusters):
        clu_center = clu['center_px']
        can_coords = {}
        for coord, coord_name in enumerate(['y', 'x', 'z']):
            if padding:
                min_ = int(clu_center[coord]-cube_shape[coord]/2)
                max_ = min_ + cube_shape[coord]
            else: 
                min_ = int(min(max(clu_center[coord]-cube_shape[coord]/2, 0), total_shape[coord]-cube_shape[coord]))
                max_ = min_ + cube_shape[coord]

            if clu['min_px'][coord] < min_ or clu['max_px'][coord] > max_:
                # print('cluster', clu_cnt, 'with center', clu['center_px'], 'too large in', coord_name, 'direction')
                if check:
                    return False
                mask_cluster = clu['mask']
                while threshold_param < 1000:
                    threshold_param *= 1.03
                    threshold = threshold_base + np.tanh(threshold_param-1)*(1-threshold_base)
                    # print(threshold, end=' ')
                    mask_threshold = weights[mask_cluster] > threshold
                    clusters_split = dbscan(X_mm[mask_cluster][mask_threshold], X_px[mask_cluster][mask_threshold], weights[mask_cluster][mask_threshold],
                                            avg_n_points_per_cmm=avg_n_points_per_cmm)
                    if split_clusters(clusters_split, cube_shape, total_shape, check=True):
                        if len(clusters_split) > 0:
                            # print('cluster', clu_cnt, 'split into', len(clusters_split), 'new clusters')
                            # print('add to center', clu['center_px'], 'new', ' '.join([str(c['center_px']) for c in clusters_split]))
                            # print('      weights', clu['prob_sum_cluster'], 'new', ' '.join([str(c['prob_sum_cluster']) for c in clusters_split]))
                            clusters_split = remove_redundant_clusters(clusters_split, cube_shape)
                            clusters_split = remove_redundant_clusters_compare_with_one(clusters_split, clu, cube_shape)
                            # print('       pruned', clu['center_px'], 'new', ' '.join([str(c['center_px']) for c in clusters_split]))
                            # print('             ', clu['prob_sum_cluster'], 'new', ' '.join([str(c['prob_sum_cluster']) for c in clusters_split]))
                            clusters_append += clusters_split
                        else:
                            # it's not so drastical as we rank clusters according to prob_sum_min_nodule_size
                            # print('cluster', clu_cnt, 'with center', clu['center_px'], 'and weight', clu['prob_sum_cluster'], 'could not be split')
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
    for clu in clusters:
        clu_center = clu['center_px']
        can_coords = {}
        for coord, coord_name in enumerate(['y', 'x', 'z']):
            if padding:
                min_ = int(clu_center[coord]-cube_shape[coord]/2)
                max_ = min_ + cube_shape[coord]
            else:
                min_ = int(min(max(clu_center[coord]-cube_shape[coord]/2, 0), total_shape[coord]-cube_shape[coord]))
                max_ = min_ + cube_shape[coord]
            can_coords['min_' + coord_name] = min_
            can_coords['max_' + coord_name] = max_
        clu['candidate_box_coords'] = can_coords
    return clusters

def get_candidates_array(clusters, array, array_name, cube_shape, threshold_prob_map=0.05):
    total_shape = array.shape
    for clu_cnt, clu in enumerate(clusters):
        coords = clu['candidate_box_coords']
        if H['resampled_lung_data_type'] != 'int16' or H['gen_prob_maps_data_type'] != 'uint8':
            raise ValueError('ERROR with datatype - create new option for the datatype you want to use')
        if array_name=='img':
            clu['candidate_'+array_name+'_array'] = np.zeros(cube_shape, dtype=np.int16)
        elif array_name=='prob_map':
            clu['candidate_'+array_name+'_array'] = np.zeros(cube_shape, dtype=np.uint8)

        clu['candidate_'+array_name+'_array'][abs(coords['min_y']) if coords['min_y']<0 else 0: cube_shape[0] if coords['max_y']<=total_shape[0] else -(coords['max_y']-total_shape[0]),
                                              abs(coords['min_x']) if coords['min_x']<0 else 0: cube_shape[1] if coords['max_x']<=total_shape[1] else -(coords['max_x']-total_shape[1]),
                                              abs(coords['min_z']) if coords['min_z']<0 else 0: cube_shape[2] if coords['max_z']<=total_shape[2] else -(coords['max_z']-total_shape[2])]\
                                                      = array[0 if coords['min_y']<0 else coords['min_y']: total_shape[0] if coords['max_y']>total_shape[0] else coords['max_y'],
                                                              0 if coords['min_x']<0 else coords['min_x']: total_shape[1] if coords['max_x']>total_shape[1] else coords['max_x'],
                                                              0 if coords['min_z']<0 else coords['min_z']: total_shape[2] if coords['max_z']>total_shape[2] else coords['max_z']]
        if array_name == 'prob_map':
            mask_threshold = clu['candidate_prob_map_array'] > threshold_prob_map * 255
            clu['prob_sum_candidate'] = np.sum(clu['candidate_prob_map_array'][mask_threshold])
    return clusters

def sort_clusters(clusters, key='prob_sum_cluster'):
    sorted_clusters = sorted(clusters, key=lambda clu: clu[key], reverse=True)
    return sorted_clusters

def remove_masks_from_clusters(clusters):
    for clu in clusters:
        del clu['mask']
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

def process_patient(patient, patient_json):
    prob_map_path = patient_json['probability_map_path']
    img_path = patient_json['resampled_lung_path']
    prob_map = np.load(prob_map_path)
    if prob_map.dtype != np.uint8:
        prob_map = (255 * prob_map).astype(np.uint8)
    prob_map_thresh = prob_map.copy()
    prob_map_thresh[prob_map_thresh < threshold_prob_map * 255] = 0.0 # here prob_map is in units of 255
    prob_map_idx = np.argwhere(prob_map_thresh)
    # translate to mm units, relative to dummy origin in pixels origin_dummy = [0, 0, 0]
    prob_map_points_mm = prob_map_idx * patient_json['resampled_scan_spacing_yxz_mm/px']

    try:
        avg_n_points_per_cmm = int(np.round(reduce(lambda x, y: x*y, [1/s for s in patient_json['resampled_scan_spacing_yxz_mm/px']])))
    except:
        wrong_spacing_warning = ('wrong resampled spacing data  {}  for patient  {}  !!!! -> continue'.format( patient_json['resampled_scan_spacing_yxz_mm/px'], patient))
        print(wrong_spacing_warning)
        logging.warning(wrong_spacing_warning)
        avg_n_points_per_cmm = 4

    prob_map_X_norm = prob_map[prob_map_idx[:, 0], prob_map_idx[:, 1], prob_map_idx[:, 2]] / 255.
    clusters = dbscan(X_mm=prob_map_points_mm, X_px=prob_map_idx, weights=prob_map_X_norm, avg_n_points_per_cmm=avg_n_points_per_cmm)
    clusters = split_clusters(clusters, cube_shape, prob_map.shape,
                              X_mm=prob_map_points_mm, X_px=prob_map_idx, weights=prob_map_X_norm, avg_n_points_per_cmm=avg_n_points_per_cmm,
                              threshold=threshold_prob_map, padding=padding_candidates)
    # print('found', len(clusters), 'clusters in', len(prob_map_points_mm), 'points')
    clusters = get_candidates_box_coords(clusters, cube_shape, prob_map.shape, padding=padding_candidates)
    clusters = get_candidates_array(clusters, prob_map, 'prob_map', cube_shape, threshold_prob_map)

    clusters = sort_clusters(clusters, key=sort_clusters_by)
    clusters = clusters[:H['max_n_candidates_per_patient']]
    clusters = remove_masks_from_clusters(clusters)

    img_array = np.load(img_path)
    clusters = get_candidates_array(clusters, img_array.copy(), 'img', cube_shape)
    # set cancer labels
    if H['data_2_process'] == 'dsb3':
        if patient in dsb3_labels['id'].values.tolist():
            patient_json['cancer_label'] = dsb3_labels['cancer'].loc[dsb3_labels['id'] == patient].tolist()[0]
        else:
            patient_json['cancer_label'] = -1
    # set nodule labels
    else:
        patient_count_nodule_prio_greater_2 = 0
        patient_json_nodules_masks = dataset_json_nodules_masks['patients'][patient]
        for clu in clusters:
            clu['nodule_priority'] = 0
            if not 'nodules' in patient_json_nodules_masks:
                continue
            nodules = patient_json_nodules_masks['nodules']
            for nodule_idx, nodule in enumerate(nodules):
                nodule_center = nodule['center_yxz_px']
                can_center = clu['center_px']
                if is_contained(can_center, nodule_center, cube_shape):
                    clu['nodule_priority'] = max(nodule['nodule_priority'], clu['nodule_priority'])
                    if clu['nodule_priority'] > 2:
                        patient_count_nodule_prio_greater_2 += 1
                        break
        patients_class = int(patient_count_nodule_prio_greater_2 > 0)
        patient_json['nodule_label'] = patients_class
    patient_json['clusters'] = [] # this is a sorted list of candidates
    patient_json['out_lst_candidates'] = []
    can_img_paths = []
    can_prob_map_paths = []
    for clu_cnt, clu in enumerate(clusters):
        patient_json['clusters'].append({})
        clu_dict = patient_json['clusters'][clu_cnt]
        # save candidate from img_array
        clu_dict['img_path'] = dataset_dir + 'candidates' + test_suffix + '/' + out_dir_name + '/' + patient + '_%02d_img.npy' % (clu_cnt)
        can_img_paths.append(clu_dict['img_path'])
        np.save(clu_dict['img_path'], clu['candidate_img_array'].astype(np.float32))
        # save candidate from prob_map
        clu_dict['prob_map_path'] = dataset_dir + 'candidates' + test_suffix + '/'  + out_dir_name + '/' + patient + '_%02d_prob_map.npy' % (clu_cnt)
        can_prob_map_paths.append(clu_dict['prob_map_path'])
        np.save(clu_dict['prob_map_path'], clu['candidate_prob_map_array'].astype(np.float32))
        # check cluster_shape
        if clu['candidate_prob_map_array'].shape != tuple(cube_shape):
            wrong_shape_error = 'WRONG shape!!! {} for patient  {}'.format(clu['candidate_prob_map_array'].shape, patient)
            print(wrong_shape_error)
            logging.error(wrong_shape_error)
            sys.exit()
        # save some cluster info
        clu_dict['prob_max_cluster'] = int(clu['prob_max_cluster']) # is all in units of 255 / int is required by json
        clu_dict['prob_sum_cluster'] = int(clu['prob_sum_cluster'])
        clu_dict['prob_sum_candidate'] = int(clu['prob_sum_candidate'])
        clu_dict['prob_sum_min_nodule_size'] = int(clu['prob_sum_min_nodule_size'])
        clu_dict['size_points_cluster'] = int(clu['size_points_cluster'])
        clu_dict['center_px'] = [int(x) for x in clu['center_px']]
        clu_dict['candidate_box_coords_yxz_px'] = [int(clu['candidate_box_coords'][k]) for k  in ['min_y', 'max_y', 'min_x', 'max_x', 'min_z', 'max_z']]
        # candidate classification info
        if H['data_2_process'] == 'LUNA16':
            patient_json['out_lst_candidates'].append('{}_{}\t{}\t{}\t{}\n'.format(patient, clu_cnt, clu['nodule_priority'], clu_dict['img_path'], clu_dict['prob_map_path']))
    patient_json['out_lst'] = '{}\t{}\t{}\t{}\n'.format(patient, patients_class, ','.join(can_img_paths), ','.join(can_prob_map_paths))
    return patient, patient_json
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('patient', nargs='?', default=None, help="provide patient id")
    parser.add_argument('--va_patients', action='store_const', const=True, default=False, help="only run on validation patients")
    parser.add_argument('--test_pipeline', type=int, required=False, default=False, help="choose num of patients processed")
    args = parser.parse_args()
    if args.patient is not None:
        print('only consider patient', args.patient)
    if args.test_pipeline:
        test_pipeline = args.test_pipeline
    elif H['test_pipeline']:
        test_pipeline = H['test_pipeline']
    else:
        test_pipeline = False
    test_suffix = '' # write to directory "'candidates' + test_suffix + '/'" for testing purposes

    sort_clusters_by = 'prob_sum_min_nodule_size'
    padding_candidates       = H['padding_candidates']
    threshold_prob_map       = H['threshold_prob_map']
    cube_shape               = H['cube_shape']

    dataset_name = H['dataset_name']
    dataset_dir = H['dataset_dir']
    dataset_json = json.load(open(dataset_dir + dataset_name + '_gen_prob_maps.json'))
    dataset_json['candidates_cube_shape'] = list(cube_shape)
    dataset_json['candidates_generation_threshold_probability_map'] = threshold_prob_map
    dataset_json['padding_candidates'] = padding_candidates

    all_patients = natsorted(list(dataset_json['patients'].keys()))
    if args.va_patients:
        try:
            dataset_json_gen_nodule_seg_lsts = json.load(open(dataset_dir + dataset_name+'_gen_nodule_seg_lsts.json'))
            print('read', dataset_dir + dataset_name+'_gen_nodule_seg_lsts.json')
        except FileNotFoundError:
            raise FileNotFoundError('Run step5_gen_nodule_seg_va_tr_lsts.py first!')
        print('only consider va_patients')
        va_patients = [patient for patient in all_patients
                       if dataset_json_gen_nodule_seg_lsts['patients'][patient]['validation_training_patient'] == 'validation']
        all_patients = va_patients

    # get patients_class for dsb3 dataset
    if H['data_2_process'] == 'dsb3':
        dsb3_labels = pd.read_csv('/'.join(H['dsb3_raw_data_in_dir'].split('/')[:-2]) + '/stage1_labels.csv')
    elif H['data_2_process'] == 'LUNA16':
        # the following is needed to get nodule positions
        try:
            dataset_json_nodules_masks = json.load(open(dataset_dir + dataset_name + '_gen_nodules+masks.json'))
            print('read', dataset_dir + dataset_name + '_gen_nodules+masks.json')
        except FileNotFoundError:
            raise FileNotFoundError('Run step4_gen_nodules+masks.py first!')
    else:
        raise ValueError('Set', H['data_2_process'], 'to either "dsb3" or "LUNA16".')
    if test_pipeline:
        all_patients = all_patients[:test_pipeline]
    n_all_patients = len(all_patients)

    out_dir_name = time.strftime('%Y_%m_%d-%H_%M', time.gmtime())
    ensure_dir(dataset_dir + 'candidates' + test_suffix + '/' + out_dir_name + '/')
    out_lst_path_patients = dataset_dir + '/candidates' + test_suffix + '/' + out_dir_name + '.lst'
    logging.basicConfig(filename=dataset_dir + '/candidates' + test_suffix + '/' + out_dir_name + '.log')
    print('saving all_candidates_lst to', out_lst_path_patients)
    # for a serial version, see the loop below
    if args.patient is None and H['gen_resampled_lung_num_threads'] > 1:
        n_jobs = min(H['gen_resampled_lung_num_threads'], len(all_patients))
        patients_candidates_dict = dict(Parallel(n_jobs=n_jobs, verbose=100)(delayed(process_patient)(patient, dataset_json['patients'][patient]) for patient in all_patients))
    out_lst_patients = open(out_lst_path_patients, 'w')
    if H['data_2_process'] == 'LUNA16':
        out_lst_path_candidates = dataset_dir + '/candidates' + test_suffix + '/' + out_dir_name + '_candidates.lst'
        out_lst_candidates = open(out_lst_path_candidates, 'w')
    for patient_cnt, patient in enumerate(tqdm(all_patients)):
        if args.patient is not None and patient != args.patient:
            continue
        if args.patient is not None or H['gen_resampled_lung_num_threads'] == 1:
            _, dataset_json['patients'][patient] = process_patient(patient, dataset_json['patients'][patient]) # serial version
        else:
            dataset_json['patients'][patient] = patients_candidates_dict[patient]
        out_lst_patients.write(dataset_json['patients'][patient]['out_lst'])
        del dataset_json['patients'][patient]['out_lst']
        if H['data_2_process'] == 'LUNA16':
            for line in dataset_json['patients'][patient]['out_lst_candidates']:
                out_lst_candidates.write(line)
            del dataset_json['patients'][patient]['out_lst_candidates']
    out_lst_patients.close()
    if H['data_2_process'] == 'LUNA16':
        out_lst_candidates.close()
    json.dump(dataset_json, open(dataset_dir + dataset_name + '_gen_candidates' + test_suffix + '.json', 'w'), indent=4)
    json.dump(dataset_json, open(dataset_dir + '/candidates' + test_suffix + '/' + out_dir_name + '.json', 'w'), indent=4) # save a copy
    print('wrote', dataset_dir + dataset_name + '_gen_candidates' + test_suffix + '.json')

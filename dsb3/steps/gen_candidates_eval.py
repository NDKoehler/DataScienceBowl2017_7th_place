"""
Compute sensitivity of candidate proposal.
"""
import os, sys
import numpy as np
import json
from collections import OrderedDict
from .. import pipeline as pipe
from . import gen_candidates

gen_nodule_masks_json = None
gen_candidates_json = None
gen_candidates_params = None
considered_patients = None

def run(max_n_candidates=20, max_dist_fraction=0.5, priority_threshold=3, 
        sort_candidates_by='prob_sum_min_nodule_size', all_patients=False):
    """
    max_n_candidates : int, optional (default: 20)
        If max_n_candidates == 0, loop over all values from 1 to 100 and return result dict
        otherwise, compute performance only for the specified value, output and save.
        In the former case, also loop over a list of max_dist_fraction.
    max_dist_fraction : float, optional (default: 0.5)
        A value of 0.5 means that the distance from the candidate center to the
        nodule center is at most 0.5 * cube_shape (accounts for each spatial
        direction), that is, the nodule center is contained in the candidate.
        Setting smaller values requires the nodule center to be closer to the
        candidate center, that is, larger parts of the nodule are required to be
        contained in the candidate.
    priority_threshold : int, optional (default: 4)
        Only consider nodules in the evaluation with a priority greater or equal to the threshold.
        Highest value 4 (4 radiologists), lowest value 1 (1 radiologist).
    sort_candidates_by : str
        # sort_candidates_by = None
        sort_candidates_by = 'prob_sum_min_nodule_size'
        # sort_candidates_by = 'prob_sum_cluster'
        # sort_candidates_by = 'prob_sum_candidate'
        # sort_candidates_by = 'size_points_cluster'
    all_patients : bool
        Consider all patients instead of only validation set.
    """

    global gen_nodule_masks_json, gen_candidates_json, gen_candidates_params, considered_patients
    gen_nodule_masks_json = pipe.load_json('out.json', 'gen_nodule_masks')
    gen_candidates_params = pipe.load_json('params.json', 'gen_candidates')
    if sort_candidates_by=='nodule_score':
        gen_candidates_json = pipe.load_json('out.json', 'filter_candidates')
    else:
	gen_candidates_json = pipe.load_json('out.json', 'gen_candidates')
    considered_patients = pipe.patients if all_patients else pipe.patients_by_split['va']
    single_patient = pipe.patients[0] if pipe.n_patients == 1 else None
    global_score = get_global_rank(sort_candidates_by, gen_candidates_json )
    if max_n_candidates > 0:
        gen_candidates_eval_json = evaluate(max_n_candidates, sort_candidates_by=sort_candidates_by, 
                                            max_dist_fraction=max_dist_fraction, single_patient=single_patient, priority_threshold=priority_threshold)
        pipe.log.debug('n_patients %s', len(considered_patients))
        pipe.log.debug('sort_candidates_by %s', gen_candidates_eval_json['sort_candidates_by'])
        pipe.log.debug('priority_threshold %s', gen_candidates_eval_json['priority_threshold'])
        pipe.log.debug('max_n_candidates %s', gen_candidates_eval_json['max_n_candidates'])
        pipe.log.debug('max_dist_fraction %s', gen_candidates_eval_json['max_dist_fraction'])
        pipe.log.debug('sensitivity %s', gen_candidates_eval_json['sensitivity'])
        pipe.log.debug('sensitivity_ignore_patient_structure %s', gen_candidates_eval_json['sensitivity_ignore_patient_structure'])
        pipe.log.debug('n_nodule_patients %s', gen_candidates_eval_json['n_nodule_patients'])
        pipe.log.debug('n_nodules %s', gen_candidates_eval_json['n_nodules'])
        pipe.log.debug('n_false_negatives %s', gen_candidates_eval_json['n_false_negatives'])
        pipe.log.debug('n_true_positives_ranked_better_than_10th %s', gen_candidates_eval_json['n_true_positives_ranked_better_than_10th'])
        pipe.log.debug('n_true_positives_ranked_better_than_20th %s', gen_candidates_eval_json['n_true_positives_ranked_better_than_20th'])
        pipe.log.debug('avg_n_redundant_candidates_per_patient %s', gen_candidates_eval_json['avg_n_redundant_candidates_per_patient'])
        pipe.log.debug('avg_deviation_from_optimal_rank %s', gen_candidates_eval_json['avg_deviation_from_optimal_rank'])
        pipe.log.debug('deviation_from_optimal_rank %s', gen_candidates_eval_json['deviation_from_optimal_rank'])
        pipe.log.debug('global_rank_score %s', global_score)
        gen_candidates_eval_json['deviation_from_optimal_rank'] = [1, 2]
        pipe.save_json('eval.json', gen_candidates_eval_json)
    else:
        filename = dataset_dir + 'evaluate/' + dataset_name+'_gen_candidates_plot_' + sort_candidates_by + '.json'
        max_dist_fraction_list = [0.5, 0.4, 0.3, 0.2]
        if not os.path.exists(filename):
            plot_json = {}
            max_n_candidates_list = list(range(5,30))
            plot_json['max_n_candidates_list'] = max_n_candidates_list 
            for max_dist_fraction in max_dist_fraction_list:
                sensitivity_list = []
                for max_n in max_n_candidates_list:
                    gen_candidates_eval_json = evaluate(max_n, sort_candidates_by=sort_candidates_by, max_dist_fraction=max_dist_fraction)
                    sensitivity_list.append(gen_candidates_eval_json['sensitivity'])
                plot_json['sensitivity_list_'+str(max_dist_fraction)] = sensitivity_list
            json.dump(plot_json, open(filename, 'w'), indent=4)
            pipe.log.debug('wrote %s', filename)
        else:
            pipe.log.debug('plotting %s', filename)
            pipe.log.debug('--> to recompute, remove this file')
            from matplotlib import pyplot as plt
            plot_json = json.load(open(filename))
            for max_dist_fraction in max_dist_fraction_list:
                plt.plot(plot_json['max_n_candidates_list'], plot_json['sensitivity_list_'+str(max_dist_fraction)], 
                         label='max_dist_fraction = '+str(max_dist_fraction))
            plt.xlabel('max_n_candidates')
            plt.ylabel('sensitivity')
            plt.xlim([0, 30])
            plt.legend()
            plt.savefig(filename.replace('.json', '.png'))


def get_global_rank(sort_candidates_by, patient_json):
    scores = []
    labels = []
    for patient_cnt, patient in enumerate(considered_patients):
        single_pat = patient_json[patient]
        for clu in single_pat['clusters']:
            nodule_rank = float(clu[sort_candidates_by])
            nodule_rank = 0 if nodule_rank<0 else nodule_rank
            nodule_priority = clu['nodule_priority']
            scores.append(nodule_rank)
            labels.append(nodule_priority)
    sorted_rank_true_positives = [x for (y,x) in sorted(zip(scores,labels), key = lambda pair: pair[0], reverse = True)]
    sorted_rank_true_positives = np.nonzero(sorted_rank_true_positives)
    ranks = [rank-i for i, rank in enumerate(sorted_rank_true_positives)]    
    #final loss values
    rank_score = np.mean(ranks)
    print("Final Sorting Score with Key: ", sort_candidates_by, "| FINAL AVG RANK: ", rank_score)
    return rank_score    

def evaluate(max_n_candidates, sort_candidates_by='prob_sum_cluster', 
             max_dist_fraction=0.5, single_patient=None, priority_threshold=4):
    false_negatives = [] # collect false negatives / non-detects
    true_positives = []
    positives = [] # all candidates
    n_fn = 0 # overall number of non-detected nodules
    n_tp = 0
    n_fp = 0
    n_true_positives_ranked_better_than_10th = 0
    n_true_positives_ranked_better_than_20th = 0
    deviation_from_optimal_rank = []
    n_nodules = 0
    sensitivities = []
    n_redundant_candidates = 0
    for patient_cnt, patient in enumerate(considered_patients):
        if single_patient is not None and patient != single_patient:
            continue
        # pipe.log.debug('patient count', patient_cnt)
        patient_json = gen_nodule_masks_json[patient]
        # pipe.log.debug('spacing_zyx_mm/px', patient_json['resampled_scan_spacing_zyx_mm/px'])
        # pipe.log.debug('origin_zyx_mm', patient_json['origin_zyx_mm'])
        # pipe.log.debug('candidates_cube_shape', dataset_json_gen_candidates['candidates_cube_shape'])
        can_cube_shape = gen_candidates_params['cube_shape']
        candidates = gen_candidates_json[patient]['clusters']
        n_candidates = len(candidates)
        n_nodules_ = 0
        n_fn_ = 0 # per patient number of non-detected nodules
        n_tp_ = 0 # per patient number of detected nodules
        n_fp_ = 0 # pet patient number of candidates that do not contain a nodule
        if not gen_nodule_masks_json[patient]['nodule_patient']:
            n_fp_ = min(n_candidates, max_n_candidates) # no candidate contains a nodule
        else: # n_nodules_ > 0
            # n_nodules_ x n_candidates list marking for each nodule in which candidate it is contained
            nodules_in_candidates = []
            nodules_indices = []
            n_nodules_ = 0
            if 'nodules' in patient_json:
                nodules = patient_json['nodules']
            else:
                pipe.log.debug('!!!!! patient %s has no key "nodules" in _gen_nodules+masks.json!!!!', patient)
                continue
            if sort_candidates_by is not None:
                candidates = gen_candidates.sort_clusters(candidates, key=sort_candidates_by)
            for nodule_idx, nodule in enumerate(nodules):
                if nodule['nodule_priority'] < priority_threshold:
                    continue
                n_nodules_ += 1
                nodules_in_candidates.append([])
                nodules_indices.append(nodule_idx)
                nodule_center = nodule['center_zyx_px']
                for ican, candidate in enumerate(candidates[:max_n_candidates]):
                    # print ('candidate center_px', candidate['center_px'])
                    can_center = candidate['center_px']
                    nodules_in_candidates[-1].append(gen_candidates.is_contained(can_center, nodule_center, can_cube_shape, max_dist_fraction))
            if n_nodules_ == 0:
                n_fp_ = min(n_candidates, max_n_candidates)
                continue
            nodules_in_candidates = np.array(nodules_in_candidates)
            # loop over nodules (rows of nodules_in_candidates)
            rank_true_positives = []
            for nodule_cnt, row in enumerate(nodules_in_candidates):
                n_candidates_per_nodule = np.sum(row)
                if n_candidates_per_nodule == 0:
                    n_fn_ += 1
                    false_negatives += [(patient, nodules_indices[nodule_cnt], '-1')]
                else:
                    n_tp_ += 1
                    rank = np.where(row)[0][0]
                    if rank < 10:
                        n_true_positives_ranked_better_than_10th += 1
                    if rank < 20:
                        n_true_positives_ranked_better_than_20th += 1                        
                    rank_true_positives += [rank]
                    true_positives += [(patient, nodules_indices[nodule_cnt], str(rank))]
                    n_redundant_candidates += np.sum(row) - 1
            # determine for each candidate, whether it contains a nodule
            for icol, col in enumerate(nodules_in_candidates.T):
                positives += [(patient, '-1', str(icol))]
                if not np.any(col):
                    n_fp_ += 1
            sensitivity_ = n_tp_ / float(n_nodules_)
            sensitivities.append(sensitivity_) # only for patients with nodules
            # some more output for non-detects
            sorted_rank_true_positives = sorted(rank_true_positives)
            print('patient', patient, 'n_nodules', n_nodules_, 'rank_true_positives', sorted_rank_true_positives, '| n_fn ', n_fn_)
            deviation_from_optimal_rank += [rank-i for i, rank in enumerate(sorted_rank_true_positives)]
            deviation_from_optimal_rank += [max_n_candidates for n in range(n_fn_)]
        n_fn += n_fn_
        n_tp += n_tp_
        n_fp += n_fp_
        n_nodules += n_nodules_

    avg_deviation_from_optimal_rank = np.mean(deviation_from_optimal_rank)
    deviation_from_optimal_rank = list(np.histogram(deviation_from_optimal_rank,
                                                    bins=np.arange(-0.5, max_n_candidates + 1.5, 1))[0].astype('int16'))
    gen_candidates_eval_json = OrderedDict()
    gen_candidates_eval_json['sort_candidates_by'] = sort_candidates_by
    gen_candidates_eval_json['priority_threshold'] = priority_threshold
    gen_candidates_eval_json['max_n_candidates'] = max_n_candidates
    gen_candidates_eval_json['max_dist_fraction'] = max_dist_fraction
    gen_candidates_eval_json['sensitivity'] = sum(sensitivities)/float(len(sensitivities))
    gen_candidates_eval_json['sensitivity_ignore_patient_structure'] = n_tp / float(n_tp + n_fn)
    gen_candidates_eval_json['avg_deviation_from_optimal_rank'] = avg_deviation_from_optimal_rank
    gen_candidates_eval_json['deviation_from_optimal_rank'] = deviation_from_optimal_rank
    gen_candidates_eval_json['n_true_positives_ranked_better_than_10th'] = n_true_positives_ranked_better_than_10th
    gen_candidates_eval_json['n_true_positives_ranked_better_than_20th'] = n_true_positives_ranked_better_than_20th
    gen_candidates_eval_json['n_nodules'] = n_nodules
    gen_candidates_eval_json['n_nodule_patients'] = len(sensitivities)
    gen_candidates_eval_json['false_negative_rate'] = n_fn / float(n_tp + n_fn) # fraction of non-detects for all nodules
    gen_candidates_eval_json['false_positive_rate'] = n_fp / float(n_tp + n_fp)
    gen_candidates_eval_json['avg_n_false_negatives_per_patient'] = n_fn / float(len(considered_patients))
    gen_candidates_eval_json['avg_n_false_positives_per_patient'] = n_fp / float(len(considered_patients))
    gen_candidates_eval_json['avg_n_redundant_candidates_per_patient'] = n_redundant_candidates / float(len(considered_patients))
    gen_candidates_eval_json['false_negatives'] = false_negatives
    gen_candidates_eval_json['n_false_negatives'] = len(false_negatives)
    gen_candidates_eval_json['true_positives'] = true_positives
    gen_candidates_eval_json['n_true_positives'] = len(true_positives)
    gen_candidates_eval_json['positives'] = positives
    return gen_candidates_eval_json


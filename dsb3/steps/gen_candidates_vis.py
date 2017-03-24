"""
Inspect prob map in the vicinity of nodule centers.
"""

import os, sys
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors
from .. import pipeline as pipe
from . import gen_candidates

def run(inspect_what='false_negatives'):
    """
    # inspect_what = 'positives'
    # inspect_what = 'false_negatives'
    # inspect_what = 'true_positives'
    """
    gen_resample_lungs_json = pipe.load_json('out.json', 'resample_lungs')
    gen_prob_maps_json = pipe.load_json('out.json', 'gen_prob_maps')
    gen_nodule_masks_json = pipe.load_json('out.json', 'gen_nodule_masks')
    gen_candidates_json = pipe.load_json('out.json', 'gen_candidates')
    gen_candidates_params = pipe.load_json('params.json', 'gen_candidates')
    gen_candidates_eval = pipe.load_json('eval.json', 'gen_candidates_eval')
    cube_shape = gen_candidates_params['cube_shape']
    threshold_prob_map = gen_candidates_params['threshold_prob_map']

    for case_cnt, (patient, nodule_cnt, cand_cnt) in enumerate(gen_candidates_eval[inspect_what]):
        print('case', case_cnt + 1, 'of', len(gen_candidates_eval[inspect_what]))
        if patient in pipe.patients:
            prob_map_path = gen_prob_maps_json[patient]['pathname']
            img_path = gen_resample_lungs_json[patient]['pathname']
            prob_map = np.load(prob_map_path)
            prob_map_thresh = prob_map.copy()
            prob_map_thresh[prob_map_thresh < threshold_prob_map*255] = 0.0

            if inspect_what in ['true_positives', 'false_negatives']:
                nodules = gen_nodule_masks_json[patient]['nodules']
                clusters = [{'center_px': nodules[nodule_cnt]['center_zyx_px']}]
                print('patient', patient, 'nodule', nodule_cnt)
                print('... nodule center', nodules[nodule_cnt]['center_zyx_px'])
            else:
                candidates = gen_candidates_json[patient]['clusters']
                candidates = gen_can.sort_clusters(candidates, key=gen_candidates_eval['sort_candidates_by'])
                candidate = candidates[int(cand_cnt)]
                clusters = [{'center_px': candidate['center_px']}]
                print('patient', patient, 'candidate', cand_cnt)
                print('... candidate center', candidate['center_px'], 'prob_sum_cluster', candidate['prob_sum_cluster'], 'prob_sum_candidate', candidate['prob_sum_candidate'])
                corresponding_nodule = [(p, n_cnt, c_cnt) for (p, n_cnt, c_cnt) in gen_candidates_eval['true_positives'] if p == patient and c_cnt == cand_cnt]
                if len(corresponding_nodule) > 0:
                    nodules = gen_nodule_masks_json[patient]['nodules']
                    nodule_cnt = corresponding_nodule[0][1]
                    print('... nodule', nodule_cnt)
                    print('... nodule center', nodules[str(nodule_cnt)]['center_zyx_px'])                
            img_array = np.load(img_path)
            clusters = gen_candidates.get_candidates_box_coords(clusters, cube_shape, prob_map.shape)
            # clusters = gen_candidates.get_candidates_array(clusters, prob_map, 'prob_map', cube_shape, prob_map.shape, padding=False)
            # clusters = gen_candidates.get_candidates_array(clusters, prob_map_thresh, 'prob_map_thresh', cube_shape, prob_map.shape, padding=False)
            # clusters = gen_candidates.get_candidates_array(clusters, img_array, 'img', cube_shape, prob_map.shape)
            clusters = get_candidates_array(clusters, prob_map, 'prob_map')
            if inspect_what == 'false_negatives':
                print('... prob_sum_around_nodule', np.sum(clusters[0]['candidate_prob_map_array']))
                candidates = gen_candidatesdidates_json['patients'][patient]['clusters']
                candidates = gen_candidates.sort_clusters(candidates, key=gen_candidates_eval['sort_candidates_by'])
                dists = [np.linalg.norm(np.array(candidate['center_px']) - np.array(nodules[nodule_cnt]['center_zyx_px'])) for candidate in candidates]                
                ican = np.argmin(dists)
                print('... closest candidate center', candidates[ican]['center_px'], 'rank', ican)
            clusters = get_candidates_array(clusters, prob_map_thresh, 'prob_map_thresh')
            clusters = get_candidates_array(clusters, img_array, 'img')
            nod_prob_map = clusters[0]['candidate_prob_map_array']
            nod_prob_map_thresh = clusters[0]['candidate_prob_map_thresh_array']
            nod_img_array = clusters[0]['candidate_img_array']
            # plotting
            plot_nodule_prob_map_img(nod_prob_map, nod_prob_map_thresh, nod_img_array)
            # plot_img_2d_slice(nod_img_array)
            plt.savefig(pipe.get_step_dir() + 'figs/' + patient + '_' + str(nodule_cnt) + '_candidate.png')
            print('plotted')
            plt.clf()
        
def plot_nodule_prob_map_img(nod_prob_map, nod_prob_map_thresh, nod_img_array):
    arrays =     [nod_prob_map, nod_prob_map_thresh,] # nod_img_array]
    thresholds = [None,         None,               ] # 0.01]
    scatters =   [True,         True,               ] # False]
    colorings =  ['Blues',      'red',              ] #  None]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors_ = plt.cm.get_cmap('jet')(plt.Normalize()(range(len(arrays))[::-1]))
    for iarray, array in enumerate(arrays):
        if scatters[iarray]:
            points = np.argwhere(array)
            indices = points[:, 0], points[:, 1], points[:, 2]            
            if colorings[iarray] == 'red':
                if len(points) == 0:
                    print('... no points with thresh_prob > 0')
                    continue
                ax.scatter(*indices, c='red', edgecolors='face', alpha=1, zorder=10, label='prob_map_thresh')
            elif colorings[iarray] == 'Blues':
                if len(points) == 0:
                    print('... no points with prob > 0')
                    continue
                print('... max prob', np.max(array[indices]) / 255., '(dtype', array.dtype, ', "* 255")') # in units of 255
                sct = ax.scatter(*indices, c=array[indices] / 255., cmap='Blues', edgecolors='face', alpha=0.3, label='prob_map')
                plt.colorbar(sct)
            else:
                print('choose colors as red or Blues')
        else:
            threshold = thresholds[iarray]
            # reduce the treshold down to a value where something can be recognized
            found_finite_values = False
            while threshold >= 0.001 or thresholds[iarray] == 0:
                try:
                    verts, faces = measure.marching_cubes(array, threshold)
                    found_finite_values = True
                    break
                except ValueError:
                    threshold *= 0.98
            if not found_finite_values:
                continue
            # Fancy indexing: `verts[faces]` to generate a collection of triangles
            mesh = Poly3DCollection(verts[faces], alpha=0.2)
            color = colors_[iarray]
            mesh.set_color(matplotlib.colors.rgb2hex(color))
            face_color = [0.5, 0.5, 1]
            mesh.set_facecolor(face_color)
            ax.add_collection3d(mesh)
    ax.set_xlim(0, arrays[0].shape[0])
    ax.set_ylim(0, arrays[0].shape[1])
    ax.set_zlim(0, arrays[0].shape[2])
    ax.set_title('plot is centered on nodule center')
    plt.legend()

def plot_img_2d_slice(array_zyx):
    z_idx = int(array_zyx.shape[0] / 2)
    plt.imshow(array_zyx[z_idx, :, :])

def get_candidates_array(clusters, array, array_name):     
    for clu in clusters:             
        coords = clu['candidate_box_coords']             
        clu['candidate_'+array_name+'_array'] = array[coords['min_y']:coords['max_y'], coords['min_x']:coords['max_x'], coords['min_z']:coords['max_z']]
    return clusters


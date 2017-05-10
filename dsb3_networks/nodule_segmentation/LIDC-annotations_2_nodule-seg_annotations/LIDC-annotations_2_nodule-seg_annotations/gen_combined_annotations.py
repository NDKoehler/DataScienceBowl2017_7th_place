import pandas as pd
import numpy as np
import json
import sys, os
from natsort import natsorted
import code
import SimpleITK as sitk
import os,sys
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
def plot_3d(nod_map): # plots annotations
    z,x,y = nod_map.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s = 40, zdir='z', c= 'red')
    # plt.show()
def plot_clusters_3d(X,y,title): # plots annotations with cluster coloring
    print(np.unique(y))
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = len(np.unique(y))
    fig = plt.figure(figsize=(6,6))
    fig.suptitle(title, fontsize=20)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for i in np.unique(y):
	       ax.scatter(X[np.where([y==i])[1], 0], X[np.where([y==i])[1], 1], X[np.where([y==i])[1], 2], color = cm(1.*i/NUM_COLORS), marker='o')
    plt.show()
def uniques_in_(string):
    unique = []
    for char in string[::]:
        if char not in unique:
            unique.append(char)
    return (len(unique))
# def dbscan(nod_map):
def dbscan(nod_map_full, yxz_to_mm): # clustering -> 2_extract_candidates.py extended to keep list index and radiologist_id
    [y_to_mm, x_to_mm, z_to_mm] = yxz_to_mm
    # only keep relevant pixels, meta info in nod_map_full
    nod_map_px = np.argwhere(nod_map_full[:,:,:,0].copy())
    nod_map = nod_map_px * yxz_to_mm
    if nod_map.shape[0] == 0:
        return []
    db = DBSCAN(eps=10.0, min_samples=1).fit(nod_map)
    if not bool(db): return []
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = np.unique(labels.astype(int))
    cluster_idx = np.zeros((nod_map.shape[0], nod_map.shape[1]+1))
    cluster_idx[:, :-1] = nod_map.copy()
    cluster_idx[:, -1] = labels
    unique_labels = np.delete(unique_labels, np.argwhere(unique_labels < 0))
    cluster_idx = cluster_idx[cluster_idx[:, -1].argsort()]
    clusters = []
    # compute cluster centers
    for clu_num in unique_labels:
        clu = {}
        clu_start = min(np.argwhere(cluster_idx[:, -1].astype(int) == clu_num)[:, 0])
        clu_end   = max(np.argwhere(cluster_idx[:, -1].astype(int) == clu_num)[:, 0])
        clu['array'] = cluster_idx[clu_start:clu_end+1, :-1]
        # attaching annotation index and radiologist_id to y,x,z coordinates
        # mm coords -> px coords to access meta info in nod_map_full via index
        cluster_points = []
        for i in clu['array']:
            point = np.append(i, nod_map_full[int(round(i[0]/y_to_mm)),int(round(i[1]/x_to_mm)),int(round(i[2]/z_to_mm)),1:])
            cluster_points.append(point)
        clu['array'] = np.array(cluster_points)
        clu['center_mm'] = [np.mean(clu['array'][:3, i].astype(np.float32)) for i in range(clu['array'].shape[1])][:3]
        clu['size_points'] = clu['array'].shape[0]
        # Number of radiologists marking this nodule
        radiologists_string = "".join([str(f) for f in list(clu['array'][:,4].astype(int))])
        clu['nodule_priority'] = int(uniques_in_(radiologists_string))
        clusters.append(clu)
    return clusters
LUNA16_annos = pd.read_csv('./LUNA16_annotations.csv', sep = ',')
LUNA16_annos['patient'] = LUNA16_annos['seriesuid'].str.split('.').str[-1]
LIDC_annos   = pd.read_csv('./LIDC_nodules_gt3mm_min1.csv', sep = '\t')
LIDC_annos['seriesuid'] = LIDC_annos['dcm_path'].str.split('/').str[-1]
LIDC_annos['patient'] = LIDC_annos['seriesuid'].str.split('.').str[-1]
# only keep LUNA16 patients
LIDC_annos = LIDC_annos.loc[LIDC_annos['seriesuid'].isin(LUNA16_annos['seriesuid'].values.tolist())]
# calculate LIDC_diameters
LIDC_annos['diameter_px'] = np.mean([LIDC_annos['x_max'].values-LIDC_annos['x_min'].values, LIDC_annos['y_max'].values-LIDC_annos['y_min'].values], axis=0)
LIDC_annos['diameter_x_px'] = LIDC_annos['x_max'].values-LIDC_annos['x_min'].values
LIDC_annos['diameter_y_px'] = LIDC_annos['y_max'].values-LIDC_annos['y_min'].values
with open('LUNA16_resample_lungs.json') as data_file:
    data = json.load(data_file)
LIDC_annos['origin_y_mm'] = [data["patients"][e]["origin_yxz_mm"][0] for e in LIDC_annos['patient']]
LIDC_annos['origin_x_mm'] = [data["patients"][e]["origin_yxz_mm"][1] for e in LIDC_annos['patient']]
LIDC_annos['origin_z_mm'] = [data["patients"][e]["origin_yxz_mm"][2] for e in LIDC_annos['patient']]
LIDC_annos['spacing_y_mm/px'] = [data["patients"][e]["raw_scan_spacing_yxz_mm/px"][0] for e in LIDC_annos['patient']] # resampled_scan_spacing_yxz_mm/px
LIDC_annos['spacing_x_mm/px'] = [data["patients"][e]["raw_scan_spacing_yxz_mm/px"][0] for e in LIDC_annos['patient']]
LIDC_annos['spacing_z_mm/px'] = [data["patients"][e]["raw_scan_spacing_yxz_mm/px"][0] for e in LIDC_annos['patient']]
LIDC_annos['nodule_id'] = [None]*len(LIDC_annos)
LIDC_annos['z_min_mm'] = [None]*len(LIDC_annos)
LIDC_annos['z_max_mm'] = [None]*len(LIDC_annos)
LIDC_annos['nodule_priority'] = [None]*len(LIDC_annos)
# LIDC_annos['center_yxz_mm'] = [None]*len(LIDC_annos)
LUNA16_patients_jsons = json.load(open('./LUNA16_resample_lungs.json'))
number_without_nodules = 0
# Iterate over patients
for pa_cnt, patient in enumerate(natsorted(LUNA16_patients_jsons['all_patients'])):
    # if pa_cnt == 4: break
    print(pa_cnt)
    patient_json = LUNA16_patients_jsons['patients'][patient]
    origin  = [float(c) for c in patient_json['origin_yxz_mm']]
    spacing = [float(c) for c in patient_json['raw_scan_spacing_yxz_mm/px']]
    LIDC_annos['origin_y_mm'].loc[LIDC_annos['patient'] == patient]  = origin[0]
    LIDC_annos['origin_x_mm'].loc[LIDC_annos['patient'] == patient]  = origin[1]
    LIDC_annos['origin_z_mm'].loc[LIDC_annos['patient'] == patient]  = origin[2]
    LIDC_annos['spacing_y_mm/px'].loc[LIDC_annos['patient'] == patient] = spacing[0]
    LIDC_annos['spacing_x_mm/px'].loc[LIDC_annos['patient'] == patient] = spacing[1]
    LIDC_annos['spacing_z_mm/px'].loc[LIDC_annos['patient'] == patient] = spacing[2]
    patient_list = LIDC_annos.loc[LIDC_annos['patient'] == patient] # list of all annotations of part. patient
    if patient_list.shape[0]==0:
        number_without_nodules +=1
        continue
    # generating 3D nodule_map with annotations = 1
    # yxz_to_mm = patient_json['resampled_scan_spacing_yxz_mm/px']
    # code.interact(local=dict(globals(), **locals()))
    yxz_to_mm = spacing
    y_max = np.max(patient_list['y_center'])
    x_max = np.max(patient_list['x_center'])
    z_max = np.max(patient_list['sliceIdx'])
    nodule_map_px = np.zeros([y_max+1, x_max+1, z_max+1, 3])
    # iterate over annotations in patient
    for idx, entry in patient_list.iterrows():
        y = patient_list.loc[[idx]]['y_center']
        x = patient_list.loc[[idx]]['x_center']
        z = patient_list.loc[[idx]]['sliceIdx']
        radiologist = patient_list.loc[[idx]]['radiologist_id']
        nodule_map_px[int(y), int(x), int(z), :] = [1, idx, radiologist]
    clusters = dbscan(nodule_map_px, yxz_to_mm)
    for cluster_nb, cluster in enumerate(clusters):
        cluster_array = cluster["array"]
        for ind in cluster_array[:,3]:
            resc = patient_json['resampled_scan_spacing_yxz_mm/px']
            LIDC_annos['nodule_id'].loc[[ind]] = int(cluster_nb)
            LIDC_annos['z_min_mm'].loc[[ind]] = np.min(cluster_array[:,2])
            LIDC_annos['z_max_mm'].loc[[ind]] = np.max(cluster_array[:,2])
            LIDC_annos['nodule_priority'].loc[[ind]] = int(cluster['nodule_priority'])
        # code.interact(local=dict(globals(), **locals()))
        ## Plots
    # if len(clusters)>3:
    #     # plot_clusters_3d(np.array([LIDC_annos['x_center'].loc[LIDC_annos['patient'] == patient], LIDC_annos['y_center'].loc[LIDC_annos['patient'] == patient], LIDC_annos['sliceIdx'].loc[LIDC_annos['patient'] == patient] ]).T, np.array(LIDC_annos['nodule_id'].loc[LIDC_annos['patient'] == patient]).T, "Clustered Nodules")
    #     # plot_3d(nodule_map[:,:,:,0])
    #     plot_clusters_3d(np.array([LIDC_annos['x_center'].loc[LIDC_annos['patient'] == patient].astype(np.int16),
    #     LIDC_annos['y_center'].loc[LIDC_annos['patient'] == patient].astype(np.int16),
    #     LIDC_annos['sliceIdx'].loc[LIDC_annos['patient'] == patient].astype(np.int16)]).T,
    #     np.array(LIDC_annos['nodule_id'].loc[LIDC_annos['patient'] == patient].astype(np.int16)).T, "Clustered Nodules")
# reconstruct coords
LIDC_annos['coordY'] = LIDC_annos['y_center']*LIDC_annos['spacing_y_mm/px'] + LIDC_annos['origin_y_mm']
LIDC_annos['coordX'] = LIDC_annos['x_center']*LIDC_annos['spacing_x_mm/px'] + LIDC_annos['origin_x_mm']
LIDC_annos['coordZ'] = LIDC_annos['sliceIdx']*LIDC_annos['spacing_z_mm/px'] + LIDC_annos['origin_z_mm']
all_annos = pd.DataFrame()
all_annos['seriesuid']   = LIDC_annos['seriesuid']
all_annos['coordY']      = LIDC_annos['coordY']
all_annos['coordX']      = LIDC_annos['coordX']
all_annos['coordZ']      = LIDC_annos['coordZ']
all_annos['nodule_id']   = LIDC_annos['nodule_id']
all_annos['nodule_priority']   = LIDC_annos['nodule_priority']
# scan spacing in raw! need it in resampled_scan_spacing_yxz_mm
# raw = 0.6mm/px, resampled = 1mm/px
all_annos['z_min_mm']    = LIDC_annos['z_min_mm'] + LIDC_annos['origin_z_mm']#to mm
all_annos['z_max_mm']    = LIDC_annos['z_max_mm'] + LIDC_annos['origin_z_mm']#to mm
# all_annos['z_min_mm']   = LIDC_annos['z_min_mm'].astype(float) + LIDC_annos['origin_z_mm'].astype(float)
# all_annos['z_max_mm']   = LIDC_annos['z_max_mm'].astype(float) + LIDC_annos['origin_z_mm'].astype(float)
# # x and y spacings are equal
all_annos['diameter_x_mm'] = LIDC_annos['diameter_x_px']*LIDC_annos['spacing_x_mm/px']
all_annos['diameter_y_mm'] = LIDC_annos['diameter_y_px']*LIDC_annos['spacing_y_mm/px']
all_annos['diameter_z_mm'] = all_annos['z_max_mm'] - all_annos['z_min_mm']
all_annos['diameter_mm'] = LIDC_annos['diameter_px'] * (LIDC_annos[['spacing_x_mm/px', 'spacing_y_mm/px']].mean(axis=1))
print("Number without nodules: ")
print(number_without_nodules)
# LIDC_annos.loc[LIDC_annos['patient']=="215640837032688688030770057224"]
all_annos.to_csv('annotations_min3.csv', index=False)
print("Done and saved")
code.interact(local=dict(globals(), **locals()))
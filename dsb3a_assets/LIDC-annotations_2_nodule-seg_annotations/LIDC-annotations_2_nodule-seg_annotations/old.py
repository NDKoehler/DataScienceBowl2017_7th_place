import pandas as pd
import numpy as np
import json
import sys, os
from natsort import natsorted
import code

# from __future__ import print_function, division
# import numpy as np
# import csv
# from glob import glob
# import pandas as pd
# import os,sys
# import cv2
import SimpleITK as sitk
# import scipy.ndimage
# from natsort import natsorted
# import dicom
# import json
# import logging
# import time
# from master_config import H
# import argparse
# import tensorflow as tf
# from tensorflow.python.training import saver as tf_saver
# from tensorflow.python.platform import resource_loader
# import multiprocessing
# from joblib import Parallel, delayed







# Task:
# - clustern der annotations in z-richtung und zusammenfuehren in einer zeile pro nodule mit den verschiedenen infos, z_min, z_max und den x/z-diamtern und der nodule_prioritaet !!!pro Schicht!!! -> viel praezisere masken
# - speichern dieser annotation_min1.csv
# - check which spacing is meant

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
LIDC_annos['z_min'] = [None]*len(LIDC_annos)
LIDC_annos['z_max'] = [None]*len(LIDC_annos)
LIDC_annos['nodule_priority'] = [None]*len(LIDC_annos)

# LUNA16_annos['origin_y_mm'] = [data["patients"][e]["origin_yxz_mm"][0] for e in LUNA16_annos['patient']]
# LUNA16_annos['origin_x_mm'] = [data["patients"][e]["origin_yxz_mm"][1] for e in LUNA16_annos['patient']]
# LUNA16_annos['origin_z_mm'] = [data["patients"][e]["origin_yxz_mm"][2] for e in LUNA16_annos['patient']]

LUNA16_patients_jsons = json.load(open('./LUNA16_resample_lungs.json'))

def getZBounds(sorted_list):
    z_boundaries = []
    z_min = 1000
    for i in sorted_list:
        if i<z_min:
            z_min=i
        if i+1 not in sorted_list:
            z_max = i
            z_boundaries.append([z_min, z_max])
            z_min = 1000
    return z_boundaries

def getPriority(sorted_list, z_min, z_max):
    radiologists = []
    for i in range(z_min, z_max):
        radiologists.extend(sorted_list["radiologist_id"].loc[LIDC_annos['sliceIdx'] == i])
    priority = len(np.unique(radiologists))
    return priority


for pa_cnt, patient in enumerate(natsorted(LUNA16_patients_jsons['all_patients'])):

    patient_json = LUNA16_patients_jsons['patients'][patient]
    origin  = [float(c) for c in patient_json['origin_yxz_mm']]
    spacing = [float(c) for c in patient_json['raw_scan_spacing_yxz_mm/px']]

    LIDC_annos['origin_y_mm'].loc[LIDC_annos['patient'] == patient]  = origin[0]
    LIDC_annos['origin_x_mm'].loc[LIDC_annos['patient'] == patient]  = origin[1]
    LIDC_annos['origin_z_mm'].loc[LIDC_annos['patient'] == patient]  = origin[2]

    LIDC_annos['spacing_y_mm/px'].loc[LIDC_annos['patient'] == patient] = spacing[0]
    LIDC_annos['spacing_x_mm/px'].loc[LIDC_annos['patient'] == patient] = spacing[1]
    LIDC_annos['spacing_z_mm/px'].loc[LIDC_annos['patient'] == patient] = spacing[2]

    uniques, counts = np.unique(LIDC_annos['sliceIdx'].loc[LIDC_annos['patient'] == patient], return_counts=True)
    # LIDC_annos.loc[LIDC_annos['patient'] == "100225287222365663678666836860"]
    nodule_id = 0
    layers_with_multiple_z_entries = [True for c in counts if c > 1]
    # loop over all z-entries:
    temp_list = LIDC_annos.loc[LIDC_annos['patient'] == patient]
    sorted_list = temp_list.sort(columns=["sliceIdx"])

    z_boundaries = (getZBounds(sorted_list["sliceIdx"].tolist()))
    for idx, nodule in enumerate(z_boundaries):
        LIDC_annos['nodule_id'].loc[LIDC_annos['patient'] == patient] == idx
        LIDC_annos['nodule_priority'].loc[LIDC_annos['patient'] == patient] == getPriority(sorted_list, nodule[0], nodule[1])
        LIDC_annos['z_min'].loc[LIDC_annos['patient'] == patient] == nodule[0]
        LIDC_annos['z_max'].loc[LIDC_annos['patient'] == patient] == nodule[1]
        code.interact(local=dict(globals(), **locals()))




        # # check multiple occurence
        # nodules_in_slice = sorted_list["sliceIdx"].tolist().count(i)
        # if nodules_in_slice>1:
        #     # handle case counts = 2
        #     print "counts"
        #     print nodules_in_slice
        #
        #     code.interact(local=dict(globals(), **locals()))


    # if True in layers_with_multiple_z_entries:
    #     temp_list["sliceIdx"]
    #     print "watch"
    #     print len([c for c in counts if c>1])
    #     print [c for c in counts if c>1]
    #     print LIDC_annos.loc[LIDC_annos['patient'] == patient].head(10)
    # else:
    #     pass

    # print(LIDC_annos['sliceIdx'].loc[LIDC_annos['patient'] == patient])



    # print (LIDC_annos['spacing_y_mm/px'].loc[LIDC_annos['patient'] == patient])
    # print (LIDC_annos['origin_y_mm'].loc[LIDC_annos['patient'] == patient])
    # print ("X-pos")
    # print (LIDC_annos['spacing_y_mm/px'].loc[LIDC_annos['patient'] == patient])
    # print(np.unique(LIDC_annos['origin_z_mm'].loc[LIDC_annos['patient'] == patient]))








    # if pa_cnt == 20: break

# reconstruct coords
LIDC_annos['coordY'] = LIDC_annos['y_center']*LIDC_annos['spacing_y_mm/px'] + LIDC_annos['origin_y_mm']
LIDC_annos['coordX'] = LIDC_annos['x_center']*LIDC_annos['spacing_x_mm/px'] + LIDC_annos['origin_x_mm']
LIDC_annos['coordZ'] = LIDC_annos['sliceIdx']*LIDC_annos['spacing_z_mm/px'] + LIDC_annos['origin_z_mm']

# nodule probability
LIDC_annos['nodule_priority'] = LIDC_annos['radiologist_id'].astype(str).str.len()

all_annos = pd.DataFrame()
all_annos['seriesuid']   = LIDC_annos['seriesuid']
all_annos['coordY']      = LIDC_annos['coordY']
all_annos['coordX']      = LIDC_annos['coordX']
all_annos['coordZ']      = LIDC_annos['coordZ']
# x and y spacings are equal
all_annos['diameter_mm'] = LIDC_annos['diameter_px']*LIDC_annos['spacing_x_mm/px']
all_annos['diameter_x_mm'] = LIDC_annos['diameter_x_px']*LIDC_annos['spacing_x_mm/px']
all_annos['diameter_y_mm'] = LIDC_annos['diameter_y_px']*LIDC_annos['spacing_y_mm/px']
all_annos['nodule_priority'] = LIDC_annos['nodule_priority']

all_annos.to_csv('annotations_min1.csv', index=False)

# test_LIDC = LIDC_annos.loc[LIDC_annos['nodule_priority']>=3].loc[LIDC_annos['coordY'].astype(str) != 'nan'].copy()
# print (test_LIDC)


# check if all LUNA16 nodules are in LIDC_annos for reconstruction validation check
# with open('test_z.lst','w') as test_lst:
#     for pa_cnt, patient in enumerate(natsorted(LUNA16_patients_jsons['all_patients'])):

#         LIDC_y = LIDC_annos['coordY'].loc[LIDC_annos['patient']==patient]
#         LUNA_y = LUNA16_annos['coordY'].loc[LUNA16_annos['patient']==patient]

#         LIDC_x = LIDC_annos['coordX'].loc[LIDC_annos['patient']==patient]
#         LUNA_x = LUNA16_annos['coordX'].loc[LUNA16_annos['patient']==patient]

#         LIDC_z = LIDC_annos['coordZ'].loc[LIDC_annos['patient']==patient]
#         LUNA_z = LUNA16_annos['coordZ'].loc[LUNA16_annos['patient']==patient]

#         test_lst.write('LIDC_z: {}, LUNA16_z: {}\n'.format(LIDC_z, LUNA_z))

#     test_lst.close()

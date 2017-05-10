import pandas as pd
import numpy as np
import sys

LUNA16_annos = pd.read_csv('./LUNA16_annotations.csv', sep = ',')
LUNA16_annos['patient'] = LUNA16_annos['seriesuid'].str.split('.').str[-1]
LIDC_annos   = pd.read_csv('./LIDC_nodules_gt3mm_min1.csv', sep = '\t')
LIDC_annos['seriesuid'] = LIDC_annos['dcm_path'].str.split('/').str[-1]
LIDC_annos['patient'] = LIDC_annos['seriesuid'].str.split('.').str[-1]

# only keep LUNA16 patients
LIDC_annos = LIDC_annos.loc[LIDC_annos['seriesuid'].isin(LUNA16_annos['seriesuid'].values.tolist())]

luna_in_lidc = set(LUNA16_annos['patient']).difference(set(LIDC_annos['patient']))

missing_patients_annos = []
for missing_pa in luna_in_lidc:
    missing_patients_anno = LUNA16_annos.loc[LUNA16_annos['patient']==missing_pa]
    missing_patients_anno['z_min_mm'] = missing_patients_anno['coordZ']-missing_patients_anno['diameter_mm']/2
    missing_patients_anno['z_max_mm'] = missing_patients_anno['coordZ']+missing_patients_anno['diameter_mm']/2
    missing_patients_anno['nodule_id'] = range(len(missing_patients_anno))

    missing_patients_annos.append(missing_patients_anno)
missing_patients_annos = pd.concat(missing_patients_annos,ignore_index=True)

missing_patients_annos.to_csv('test.csv', ignore_index=True)
print (len(missing_patients_annos))

all_nodules = []
eps = 0.1
for nodule_cnt in range(len(missing_patients_annos)):
    nodule = missing_patients_annos.loc[nodule_cnt]
    diam_mm = nodule['diameter_mm']
    for layer_mm in np.linspace(-diam_mm/2, diam_mm/2, diam_mm): # mm steps
        nodule_layers_DF = {}

        nodule_layers_DF['seriesuid'] = nodule['seriesuid']
        nodule_layers_DF['nodule_id'] = nodule['nodule_id']

        nodule_layers_DF['z_min_mm'] = nodule['z_min_mm']
        nodule_layers_DF['z_max_mm'] = nodule['z_max_mm']

        nodule_layers_DF['coordZ'] = nodule['z_min_mm'] + layer_mm + diam_mm/2
        nodule_layers_DF['coordX'] = nodule['coordX']
        nodule_layers_DF['coordY'] = nodule['coordY']

        nodule_layers_DF['diameter_mm'] = nodule['diameter_mm']

        xy_diam_mm = 2*np.sqrt(np.abs(((diam_mm/2+eps)**2-layer_mm**2)))
        nodule_layers_DF['diameter_x_mm'] = xy_diam_mm
        nodule_layers_DF['diameter_y_mm'] = xy_diam_mm
        nodule_layers_DF['diameter_z_mm'] = diam_mm

        nodule_layers_DF['nodule_priority'] = 3

        nodule_layers_DF = pd.DataFrame(pd.DataFrame(nodule_layers_DF, index=[0]))
        all_nodules.append(nodule_layers_DF)

all_nodules = pd.DataFrame(pd.concat(all_nodules, ignore_index=True))
print ('len all_missing_nodules_layers', len(all_nodules))

all_nodules.to_csv('missing_patients.csv', index=False)

print ('len LUNA: {}, len luna not in lidc {}'.format(len(set(LUNA16_annos['patient'])), len(luna_in_lidc)))

# combined lists
annotations = pd.read_csv('./annotations_min.csv')
print (len(annotations))
annotations = pd.concat([annotations,all_nodules])
print (len(annotations))

annotations.to_csv('./annotations_min+missing_LUNA16_patients.csv', ignore_index=True)

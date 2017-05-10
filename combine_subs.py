import pandas as pd
import numpy as np
import os
from dsb3 import utils

outpath = './out/'
utils.ensure_dir(outpath)

data1 = pd.read_csv('./datapipeline_final/dsb3_0/gen_submission_2D_05res_80/submission.csv')
data2 = pd.read_csv('./datapipeline_final/dsb3_0/gen_submission_2D_07res_80/submission.csv')
data3 = pd.read_csv('./datapipeline_final/dsb3_0/gen_submission_3D_05res_80/submission.csv')
data4 = pd.read_csv('./datapipeline_final/dsb3_0/gen_submission_3D_07res_80/submission.csv')

data1['cancer'] += data2['cancer']
data1['cancer'] += data3['cancer']
data1['cancer'] += data4['cancer']
data1['cancer'] /= 4

data1.to_csv(outpath + 'submission_80.csv', index = False)
data1 = pd.read_csv('./datapipeline_final/dsb3_0/gen_submission_2D_05res_100/submission.csv')
data2 = pd.read_csv('./datapipeline_final/dsb3_0/gen_submission_2D_07res_100/submission.csv')
data3 = pd.read_csv('./datapipeline_final/dsb3_0/gen_submission_3D_05res_100/submission.csv')
data4 = pd.read_csv('./datapipeline_final/dsb3_0/gen_submission_3D_07res_100/submission.csv')

data1['cancer'] += data2['cancer']
data1['cancer'] += data3['cancer']
data1['cancer'] += data4['cancer']
data1['cancer'] /= 4

data1.to_csv(outpath + 'submission_100.csv', index = False)

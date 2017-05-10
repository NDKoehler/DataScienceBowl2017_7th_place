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
import math
import code


import matplotlib.pyplot as plt
import numpy as np

annotations = pd.read_csv("annotations_min4.csv") # its possible to use more precise nodule information from candidates_V2.csv

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

# annotations.dropna()

def plot_histo(X, title):
    plt.hist(X, bins=4)
    plt.title(title)
    plt.xlabel("Priority")
    plt.xlim([1,4])
    plt.ylabel("Frequency")
    plt.show()

def get_stats(X, title):
    print title
    print "Mean: ", np.mean(X)
    print "Std: ", np.std(X)
    print "Total: ", len(X)
    print "Min: ", np.min(X)
    print "Max: ", np.max(X)


LUNA16_annos = pd.read_csv('./LUNA16_annotations.csv', sep = ',')
LUNA16_annos['patient'] = LUNA16_annos['seriesuid'].str.split('.').str[-1]

np.unique(annotations["nodule_priority"], return_counts=True)
# indices = []
# for i in xrange(annotations.shape[0]):
#     if math.isnan(annotations["nodule_priority"].ix[i]):
#         print annotations["seriesuid"].ix[i][34:]
#         print annotations.ix[i]
        # print LUNA16_annos.loc[LUNA16_annos['patient']==annotations["seriesuid"].ix[i][34:]]

#         # .split('.').str[-1]
#
# print indices
# get_stats(annotations["nodule_priority"].dropna(), "Nodule Priorities")
code.interact(local=dict(globals(), **locals()))
LUNA16_annos.loc[LUNA16_annos['patient']=="215640837032688688030770057224"]

# Failure at:
# [162, 166, 639, 863, 1441, 1827, 1829, 1830, 1863, 3164, 3286, 3499, 3762, 3834, 3836, 3860, 3868, 3874, 3983, 3989, 3998, 4000, 4001, 4269, 5177, 5610, 6048, 6055, 6063, 6075, 6205, 6218, 6418, 6844, 6920, 6926, 7470, 8272, 9413, 9894, 9902, 10156, 10197, 11531, 11532, 11533, 11534, 11535, 11540, 11827, 11830, 12077, 12078, 12079, 12110]



# plot_histo(annotations["nodule_priority"].dropna().astype(int), "Nodule Priority")

import pandas as pd
import numpy as np
import sys 

#Script for reordering training list to the identical ordering of the 7April submission (reproducibility)


data = pd.read_csv('./dsb3a_assets/patients_lsts/dsb3/enforce_ordering/tr_patients_100.lst', header=None, sep = '\t')
data_new = pd.read_csv('./datapipeline_final/dsb3_0/interpolate_candidates_res05/tr_patients_100.lst', header=None, sep = '\t')

data.set_index([0], inplace = True)
data_new.set_index([0], inplace = True)
data_new = data_new.loc[data.index]

data_new.to_csv('./datapipeline_final/dsb3_0/interpolate_candidates_res05/tr_patients_100.lst', header=None, sep = '\t')

#check
data_new = pd.read_csv('./datapipeline_final/dsb3_0/interpolate_candidates_res05/tr_patients_100.lst', header=None, sep = '\t')[0].values.tolist()
data = pd.read_csv('./dsb3a_assets/patients_lsts/dsb3/enforce_ordering/tr_patients_100.lst', header=None, sep = '\t')[0].values.tolist()

for x,y in zip(data, data_new):
    if x != y:
        print("enforce ordering failed")
        print(x, y)
        sys.exit()    



data = pd.read_csv('./dsb3a_assets/patients_lsts/dsb3/enforce_ordering/tr_patients_100.lst', header=None, sep = '\t')
data_new = pd.read_csv('./datapipeline_final/dsb3_0/interpolate_candidates_res07/tr_patients_100.lst', header=None, sep = '\t')

data.set_index([0], inplace = True)
data_new.set_index([0], inplace = True)
data_new = data_new.loc[data.index]

data_new.to_csv('./datapipeline_final/dsb3_0/interpolate_candidates_res07/tr_patients_100.lst', header=None, sep = '\t')



#check
data_new = pd.read_csv('./datapipeline_final/dsb3_0/interpolate_candidates_res07/tr_patients_100.lst', header=None, sep = '\t')[0].values.tolist()
data = pd.read_csv('./dsb3a_assets/patients_lsts/dsb3/enforce_ordering/tr_patients_100.lst', header=None, sep = '\t')[0].values.tolist()

for x,y in zip(data, data_new):
    if x != y:
        print("enforce ordering failed")
        print(x, y)
        sys.exit()    


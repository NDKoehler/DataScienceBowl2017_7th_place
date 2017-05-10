import pandas as pd
import numpy as np

xml_paths = np.genfromtxt("xml_path_lst_unblinded_read.csv",dtype=str)

patient_id_lst = []
for xml_path in xml_paths:
  patient_id_lst += [xml_path.split("/")[7]]
  
print len(set(patient_id_lst))
print len(patient_id_lst)
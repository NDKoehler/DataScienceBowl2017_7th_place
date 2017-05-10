import os,sys
import numpy
import SimpleITK
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

         
def sitk_show_opencv(slice_array, nodule_position_lst):
    
    diff = np.max(slice_array) - np.min(slice_array)
    slice_array = slice_array - np.min(slice_array)
    slice_array = slice_array / float(diff)
    
    for nodule_position in nodule_position_lst:
      cv2.rectangle(slice_array,(nodule_position[0]-5,nodule_position[1]-5),(nodule_position[0]+5,nodule_position[1]+5), (255,0,0),1)
      
    cv2.imshow("view nodule",slice_array)
    cv2.waitKey(0)


    
nodule_info_df = pd.read_csv("/media/philipp/qnap/LIDC/preprocessing/4_get_nodule_info_per_slice/dataframe_nodules_gt3mm.csv",sep="\t")
xml_paths_unblinded_read = np.genfromtxt("/media/philipp/qnap/LIDC/preprocessing/2_get_xml_paths_unblinded_read/xml_path_lst_unblinded_read.csv", dtype=str)
original_spacing_df = pd.read_csv("/media/philipp/qnap/LIDC/preprocessing/3_save_dicom_metadata/dicom_metadata.csv",header=0,sep="\t")

min_experts = 1
max_dist = 5 # max_distance in millimeters



verified_nodules_radiologist_id = []
verified_nodules_nodule_id = []
verified_nodules_dcm_path = []
verified_nodules_x_center = []
verified_nodules_y_center = []
verified_nodules_sliceIdx = []
verified_nodules_x_min = []
verified_nodules_x_max = []
verified_nodules_y_min = []
verified_nodules_y_max = []

for counter, xml_path in enumerate(xml_paths_unblinded_read):
  print (counter+1,len(xml_paths_unblinded_read))
  
  dcm_path = os.path.dirname(os.path.abspath(xml_path))
  dcm_path = dcm_path.split('/')
  dcm_path[2] = "philipp"
  dcm_path = '/'.join(dcm_path)

  this_spacing_df = original_spacing_df[original_spacing_df['dcm_path'] == dcm_path]
  
  if(len(this_spacing_df) != 1): # if dcm_path does not exist in dcm_path_df: maxbe wrong username?
    print "dcm_path not found in /media/philipp/qnap/LIDC/preprocessing/3_write_original_spacing_info/original_spacings.csv"
    print "wrong username?"
    sys.exit()
  
  x_spacing = this_spacing_df["x_spacing"].values[0]
  y_spacing = this_spacing_df["y_spacing"].values[0]
  epsilon = int(max_dist/np.max(x_spacing,y_spacing))

  
  
  nodules_in_dcm = nodule_info_df[nodule_info_df["dcm_path"] == dcm_path]
  sliceIdx_set = list(set(nodules_in_dcm["sliceIdx"].values))
  
  sliceIdx_set = [x for x in sliceIdx_set if x >= 0] # delete negative slice Ids (warum existieren die???)
  
  for sliceIdx in sliceIdx_set:
    nodules_in_slice = nodules_in_dcm[nodules_in_dcm["sliceIdx"] == sliceIdx]
    
    
    radiologist_id_arr = nodules_in_slice["radiologist_id"].values
    
    x_center_arr = nodules_in_slice["x_center"].values
    y_center_arr = nodules_in_slice["y_center"].values
    
    x_min_arr = nodules_in_slice["x_min"].values
    x_max_arr = nodules_in_slice["x_max"].values

    y_min_arr = nodules_in_slice["y_min"].values
    y_max_arr = nodules_in_slice["y_max"].values
     
    
    nodule_positions = np.asarray(zip(x_center_arr, y_center_arr))
    db = DBSCAN(eps=epsilon, min_samples=min_experts).fit(nodule_positions)
    labels = db.labels_

    for cluster_id in list(set(labels)):
      if(cluster_id != -1):
	
	cluster_nodules_radiologist_id = radiologist_id_arr[labels == cluster_id]

	cluster_nodules_x_center = x_center_arr[labels == cluster_id]
	cluster_nodules_y_center = y_center_arr[labels == cluster_id]
	
	cluster_nodules_x_min = x_min_arr[labels == cluster_id]
	cluster_nodules_x_max = x_max_arr[labels == cluster_id]
	
	cluster_nodules_y_min = y_min_arr[labels == cluster_id]
	cluster_nodules_y_max = y_max_arr[labels == cluster_id]
	
	if(len(set(cluster_nodules_radiologist_id)) >= min_experts ): #check ob alle markierungen tatsaechlich von UNTERSCHIEDLICHEN radiologen kommen!
	
	  string = ""
	  for rad in cluster_nodules_radiologist_id:
	    string += str(rad)
	  
	  verified_nodules_radiologist_id += [string]	  
	  verified_nodules_nodule_id += ["merged"]
	  verified_nodules_dcm_path += [dcm_path]
	  verified_nodules_x_center += [int(np.mean(cluster_nodules_x_center))]
	  verified_nodules_y_center += [int(np.mean(cluster_nodules_y_center))]
	  verified_nodules_sliceIdx += [sliceIdx]
	  verified_nodules_x_min += [np.min(cluster_nodules_x_min)]
	  verified_nodules_x_max += [np.max(cluster_nodules_x_max)]
	  verified_nodules_y_min += [np.min(cluster_nodules_y_min)]
	  verified_nodules_y_max += [np.max(cluster_nodules_y_max)]
	  
	  #print len(verified_nodules_radiologist_id), len(verified_nodules_nodule_id), len(verified_nodules_x_min)
	  #print '---------------------------------------------------'





print len(verified_nodules_radiologist_id)
df = pd.DataFrame()
df.insert(0,"radiologist_id",verified_nodules_radiologist_id)
df.insert(1,"nodule_id",verified_nodules_nodule_id)
df.insert(2,"dcm_path",verified_nodules_dcm_path)
df.insert(3,"x_center",verified_nodules_x_center)
df.insert(4,"y_center",verified_nodules_y_center)
df.insert(5,"sliceIdx",verified_nodules_sliceIdx)
df.insert(6,"x_min",verified_nodules_x_min)
df.insert(7,"x_max",verified_nodules_x_max)
df.insert(8,"y_min",verified_nodules_y_min)
df.insert(9,"y_max",verified_nodules_y_max)

df.to_csv('nodules_gt3mm_min'+str(min_experts)+'.csv', sep = '\t')

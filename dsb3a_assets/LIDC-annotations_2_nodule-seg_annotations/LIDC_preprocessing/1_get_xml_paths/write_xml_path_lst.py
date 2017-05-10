import os,sys
import numpy as np

doi_root_path = "/media/philipp/qnap/LIDC/download/DOI/"

lidc_idri_folders = []
study_instances = []
series_instances = []
xml_path_lst = []

for doi_folder in os.listdir(doi_root_path):
  lidc_idri_folder = doi_root_path+doi_folder+"/"
  lidc_idri_folders += [lidc_idri_folder]
  for lidc_folder in os.listdir(lidc_idri_folder):
    study_instance = lidc_idri_folder+lidc_folder+"/"
    study_instances += [study_instance]
    for study_folder in os.listdir(study_instance):
      series_instance = study_instance+study_folder+"/"
      
      for file in os.listdir(series_instance): # check if folder contains valid xml file
	if(file.endswith(".xml")):
          series_instances += [series_instance]
          xml_path_lst += [series_instance+file]
	  break

print len(xml_path_lst)
np.savetxt("xml_path_lst.csv", xml_path_lst,fmt="%s")

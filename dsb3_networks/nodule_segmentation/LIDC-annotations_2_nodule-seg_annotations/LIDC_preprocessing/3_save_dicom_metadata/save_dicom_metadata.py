import os
import SimpleITK
import pandas as pd
import cv2
import numpy as np
import sys


xml_paths = np.genfromtxt("/media/philipp/qnap/LIDC/preprocessing/2_get_xml_paths_unblinded_read/xml_path_lst_unblinded_read.csv",dtype=str,delimiter=",")



dcm_path_lst = []
offset_x_lst = []
offset_y_lst = []
offset_z_lst = []

spacing_x_lst = []
spacing_y_lst = []
spacing_z_lst = []

dimsize_x_lst = []
dimsize_y_lst = []
dimsize_z_lst = []

counter = 1
for xml_path in xml_paths:
  print (counter,len(xml_paths))
  counter += 1

  dcm_path = os.path.dirname(os.path.abspath(xml_path))
  reader = SimpleITK.ImageSeriesReader()
  filenamesDICOM = reader.GetGDCMSeriesFileNames(dcm_path)
  reader.SetFileNames(filenamesDICOM)
  imgOriginal = reader.Execute()
  
  dimSize_x = imgOriginal.GetWidth()
  dimSize_y = imgOriginal.GetHeight()
  dimSize_z = imgOriginal.GetDepth()
  dimSize = (dimSize_x,dimSize_y,dimSize_z)
  offset = imgOriginal.GetOrigin()
  spacing = imgOriginal.GetSpacing()

  dcm_path_lst += [dcm_path]

  offset_x_lst += [offset[0]]
  offset_y_lst += [offset[1]]
  offset_z_lst += [offset[2]]

  spacing_x_lst += [spacing[0]]
  spacing_y_lst += [spacing[1]]
  spacing_z_lst += [spacing[2]]
  
  dimsize_x_lst += [dimSize[0]]
  dimsize_y_lst += [dimSize[1]]
  dimsize_z_lst += [dimSize[2]]
  
df = pd.DataFrame()
df.insert(0,"dcm_path",dcm_path_lst)
df.insert(1,"x_spacing",spacing_x_lst)
df.insert(2,"y_spacing",spacing_y_lst)
df.insert(3,"z_spacing",spacing_z_lst)
df.insert(4,"x_offset",offset_x_lst)
df.insert(5,"y_offset",offset_y_lst)
df.insert(6,"z_offset",offset_z_lst)
df.insert(7,"x_dimsize",dimsize_x_lst)
df.insert(8,"y_dimsize",dimsize_y_lst)
df.insert(9,"z_dimsize",dimsize_z_lst)
df.to_csv('dicom_metadata.csv', sep = '\t',index = False)
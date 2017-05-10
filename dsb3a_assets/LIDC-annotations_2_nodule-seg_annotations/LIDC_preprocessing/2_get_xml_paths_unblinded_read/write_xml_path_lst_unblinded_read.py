import os,sys
import numpy as np
import xml.etree.ElementTree as ET
import SimpleITK


xml_path_lst = np.genfromtxt('/media/philipp/qnap/LIDC/preprocessing/1_get_xml_paths/xml_path_lst.csv', delimiter=",",dtype=str)
xml_path_lst_unblinded_read = []
min_slice_thickness = 2.5

counter = 1
for xml_path in xml_path_lst:
  print (counter, len(xml_path_lst))
  counter += 1
  tree = ET.parse(xml_path)
  root = tree.getroot()

  if(len(root.attrib.keys()) == 1):
    print xml_path +  "  -  incorrect header"
    continue # falls header nciht sauber geschrieben wurde ("," fehlt)

  if(root.attrib.keys()[0] != "uid"):
    prefix =  root.attrib[ root.attrib.keys()[0] ].split()[0]
  else:
    prefix =  root.attrib[ root.attrib.keys()[1] ].split()[0]

  TaskDescription = root[0].findall('{'+prefix+'}TaskDescription')[0].text


  if(TaskDescription == "Second unblinded read"):
  
  
    dcm_path = os.path.dirname(os.path.abspath(xml_path))
    
    
    reader = SimpleITK.ImageSeriesReader()
    filenamesDICOM = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(filenamesDICOM)
    imgOriginal = reader.Execute()
    spacing = imgOriginal.GetSpacing()
    
    if(spacing[2] <= min_slice_thickness):
      xml_path_lst_unblinded_read += [xml_path]


print len(xml_path_lst_unblinded_read)
np.savetxt("xml_path_lst_unblinded_read.csv", xml_path_lst_unblinded_read,fmt="%s")

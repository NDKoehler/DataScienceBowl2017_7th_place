import numpy as np
import os,sys
import cv2
import json
import fnmatch
from tqdm import tqdm

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def main(json_path, out_folder_path):
    json_file= json.load(open(json_path))
    for cnt, element in enumerate(tqdm(json_file)):
        print (cnt)
        filename = element['filename']

        # print(json_path)
        print (filename)

        annotations = element['annotations']
        if len(annotations) == 0:
            continue

        img = cv2.imread(filename,0)
        mask = np.zeros_like(img, np.uint8)


        # annot_image = img.copy()
        for an in annotations:
            try:
                x_coos = map(float, an['xn'].split(';'))
                y_coos = map(float, an['yn'].split(';'))
                points = zip(x_coos, y_coos)
                points = [np.array(points, dtype=np.int32)]

                # cv2.drawContours(annot_image,points,-1,(255),-1)
                cv2.drawContours(mask,points,-1,(255,255,255),-1)
            except:
                print ('\nERROR with file ', filename)
                print (an)

        # cv2.imshow("asdf", cv2.resize(annot_image, (512,512)))

        if 1:
            out_path = out_folder_path + filename.split('/')[-1].split('.jpg')[0]

            image_path =  out_path + '.jpg'
            mask_path  =  out_path + '_mask.jpg'


            cv2.imwrite(image_path, img)
            cv2.imwrite(mask_path, mask)
            # cv2.waitKey(0)


if __name__ == '__main__':
    json_path = '../data/annotations/anno_2.json'
    out_folder_path = '../data/annotations/lung_wings_imgs/'
    ensure_dir(out_folder_path)
    main(json_path, out_folder_path)


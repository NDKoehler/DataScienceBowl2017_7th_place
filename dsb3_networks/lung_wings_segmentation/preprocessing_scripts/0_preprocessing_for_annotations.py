from __future__ import print_function, division

import dicom
import numpy as np
import glob
import os,sys
import cv2
import json
from natsort import natsorted
import multiprocessing
from joblib import Parallel, delayed
import SimpleITK as sitk
import scipy.ndimage
from tqdm import tqdm

np.random.seed(17) # do NOT change

def get_img_array_dcom(img_file):
    def load_scan(path):
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
        return slices
    def get_pixels_hu(slices):
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)
        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)
        return np.array(image, dtype=np.int16)
    scan = load_scan(img_file)
    try:
        acquisition_numbers = [int(s.AcquisitionNumber) for s in scan]
    except:
        acquisition_numbers = [None for s in scan]
        print('no acquisition_numbers for {}'.format(img_file))

    img_array = get_pixels_hu(scan) # z,y,x
    spacing = np.array(list(map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing)))) # z,y,x
    return img_array, spacing, None, acquisition_numbers

def resample(pa_lst, target_spacing, data_type):
    image, spacing, origin, original_shape, acquisition_numbers, patient = pa_lst

    # Determine current pixel spacing
    resize_factor = spacing / np.array(target_spacing)
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    real_spacing = np.array(spacing) / np.array(real_resize_factor)
    # 3d interpolation
    image = scipy.ndimage.interpolation.zoom(image if data_type=='int16' else image.astype(np.float32), real_resize_factor, order=2, mode='nearest')
    return [image, spacing, real_spacing, origin, original_shape, acquisition_numbers, patient]

def clip_HU_range(image,HU_tissue_range):
    # tissue range [-1000, 400]
    image = image - HU_tissue_range[0]
    image[image > (HU_tissue_range[1]-HU_tissue_range[0]) ] = (HU_tissue_range[1]-HU_tissue_range[0])
    image[image<0] = 0
    return image.astype(np.int16)

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def main(output_img_dir, output_anno_dir, npys, num_threads, num_rand_imgs_per_patient, HU_tissue_range):

    all_junks = []
    num_junks = int(np.ceil(len(npys) / num_threads ) )
    for junk_cnt in range(num_junks):
        junk = []
        for in_junk_cnt in range(num_threads):
            patient_cnt = num_threads*junk_cnt + in_junk_cnt
            # break after last patient
            if patient_cnt >= len(npys): break

            junk.append(npys[patient_cnt])
        all_junks.append(junk)

    annot_lst = []
    embed = np.zeros((512,512),dtype=np.uint8)
    # loop over junks
    for junk_cnt, junk in enumerate(tqdm(all_junks)):
        img_array_junk = []
        for npy in junk:
            # file type == dcom!!!
            img_array, spacing, origin, acquisition_numbers = get_img_array_dcom(npy)
            original_shape = img_array.shape
            img_array_junk.append([img_array, spacing, origin, original_shape, acquisition_numbers, npy.split('/')[-2]]) # z,y,x
        #--------------------------------multithread-------------------------------------
        # heterogenous spacing -> homogeneous spacing
        resampled_junk_lst = Parallel(n_jobs=min([num_threads,len(junk)]))(delayed(resample)(pa_lst, [1,1,1], data_type='int16') for pa_lst in img_array_junk)
        #-------------------------------multithread-------------------------------------
        for pa_cnt, pa_lst in enumerate(resampled_junk_lst):
            img = pa_lst[0]
            for rand_lay in np.random.permutation(range(img.shape[0]))[:num_rand_imgs_per_patient]:
                embed[:] = 0

                lay = (clip_HU_range(img[rand_lay,:,:].copy(), HU_tissue_range)/1400.*255).astype(np.uint8)

                # .astype(np.uint8)
                if len(lay.shape) > 2:
                    np.squeeze(lay,2)
                embed[(embed.shape[0] - lay.shape[0])//2:(embed.shape[0] - lay.shape[0])//2 + lay.shape[0],(embed.shape[1] - lay.shape[1])//2:(embed.shape[1] - lay.shape[1])//2 + lay.shape[1]] = lay
                output_path = output_img_dir + pa_lst[-1] + '_lay_{}.jpg'.format(rand_lay)
                cv2.imwrite(output_path, embed)
                annot_lst.append({'filename':output_path,'annotations':[],'class':'image'})
        json.dump(annot_lst, open(output_anno_dir + 'anno_2.json','w'), indent=4)
if __name__ == '__main__':
    raw_dsb3_path = '/media/juler/qnap/DATA/dsb3/stage1/*/'
    npys = np.random.permutation(glob.glob(raw_dsb3_path))[:30]
    output_img_dir = '../data/raw_imgs/'
    output_anno_dir = '../data/annotations/'
    ensure_dir(output_anno_dir)
    ensure_dir(output_img_dir)
    num_threads = 6
    num_rand_imgs_per_patient = 5
    HU_tissue_range = [-1000,400]

    main(output_img_dir, output_anno_dir, npys, num_threads, num_rand_imgs_per_patient, HU_tissue_range)


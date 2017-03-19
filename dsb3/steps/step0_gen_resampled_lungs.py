from __future__ import print_function, division
import numpy as np
import csv
from glob import glob
import pandas as pd
import os,sys
import cv2
import SimpleITK as sitk
import scipy.ndimage
from natsort import natsorted
import dicom
import json
import logging
import time
from master_config import H
# from dsb3_data_except_dict import mlt_acquisitions_dict
import argparse
if __name__ == '__main__':
    # avoid allocating memory with tensorflow when importing from step3
    import tensorflow as tf
    from tensorflow.python.training import saver as tf_saver
    from tensorflow.python.platform import resource_loader
    lung_wings_seg_checkpoint_dir = H['lung_wings_seg_checkpoint_dir']
    sys.path.append(lung_wings_seg_checkpoint_dir+'/architecture/')
    from basic_logging import initialize_logger
    import unet_model as lung_wings_unet_model
import multiprocessing
from joblib import Parallel, delayed
from utils import ensure_dir
import math
import code

try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x
def get_pre_normed_value_hist(img_array):
    hist,ran = np.histogram(img_array.flatten(), bins=16*5,normed=True,range=[-1000,600])
    return hist, ran
def get_img_array_mhd(img_file):
    # load the data once
    itk_img = sitk.ReadImage(str(img_file))
    img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
    # no acquisition number found in object
    original_shape = img_array.shape
    original_shape = [original_shape[2], original_shape[1], original_shape[0]] # y,x,z
    origin = itk_img.GetOrigin()      # x,y,z  Origin in world coordinates (mm)
    origin = [float(origin[1]), float(origin[0]), float(origin[2])] # y,x,z
    spacing = itk_img.GetSpacing()    # spacing of voxels in world coor. for x,y,z (mm)
    spacing = np.array([spacing[2],spacing[1],spacing[0]]) # z,y,x
    return img_array, spacing, origin, None

def get_img_array_dcom(img_file):
    def load_scan(path):
        patient = path.split("/")[-2]
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        unique_ac_nums, counts = np.unique([s.AcquisitionNumber for s in slices], return_counts = True)
        if len(unique_ac_nums) > 1:
            counts = [int(i) for i in counts]
            print("Multiple scan exception, different acquisition numbers: {}".format(unique_ac_nums))
            print("Counts: {}".format(counts))
            print("Patient {}".format(patient))
            # Selecting the index with the highest number of acquisitions. In case of balanced acquisitions,
            # selecting the latter. Operation is string compatible
            selected_acquisition = unique_ac_nums[np.argwhere(counts == np.amax(counts))[-1][0]]
            print("proceding with most frequent acquisition number {}".format(selected_acquisition))
            slices = [s for s in slices if s.AcquisitionNumber == selected_acquisition]

            acquisition_exception_json = {}
            multiple_scan_exception = [[str(i) for i in unique_ac_nums], counts]
            acquisition_exception_json["multiple_scan_exception_uniques/counts"] = multiple_scan_exception
            acquisition_exception_json["selected_acquisition"] = str(selected_acquisition)

        elif len(unique_ac_nums) == 0:
            acquisition_exception_json = "No AcquisitionNumber"
            print("Patient {} without acquisition number.".format(patient))
        else:
            acquisition_exception_json = None
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        spacings = [s.PixelSpacing for s in slices]
        for s in slices:
            s.SliceThickness = slice_thickness
            # Check spacing
            for i in range(2):
                if math.isnan(s.PixelSpacing[i]) or s.PixelSpacing[i] < 0.1 or s.PixelSpacing[i] > 5 or isinstance(s.PixelSpacing[i], str): # Check for exceptions
                    # clean list of exceptions before taking mode.
                    cleaned_spacings = [sp[i] for sp in spacings if not math.isnan(sp[i]) if not sp[i] < 0.1 if not sp[i] > 5 if not isinstance(sp[i], str)]
                    # Take the mode value of spacings, or the first value if values are even.
                    s.PixelSpacing[i]=np.argmax(np.bincount(cleaned_spacings))
        return slices, acquisition_exception_json
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
    scan, acquisition_exception_json = load_scan(img_file)
    # try:
    #     acquisition_numbers = [int(s.AcquisitionNumber) for s in scan]
    # except:
    #     acquisition_numbers = [None for s in scan]
    #     logging.warning('no acquisition_numbers for {}'.format(img_file))

    img_array = get_pixels_hu(scan) # z,y,x
    spacing = np.array(list(map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing)))) # z,y,x
    return img_array, spacing, None, acquisition_exception_json

def resample(pa_lst, target_spacing, data_type):
    image, spacing, origin, original_shape, acquisition_exception_json, patient = pa_lst

    # Determine current pixel spacing
    resize_factor = spacing / np.array(target_spacing)
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    real_spacing = np.array(spacing) / np.array(real_resize_factor)
    # 3d interpolation
    image = scipy.ndimage.interpolation.zoom(image if data_type=='int16' else image.astype(np.float32), real_resize_factor, order=3, mode='nearest')
    return [image, spacing, real_spacing, origin, original_shape, acquisition_exception_json, patient]

def clip_HU_range(image,HU_tissue_range):
    # tissue range [-1000, 400]
    image = image - HU_tissue_range[0]
    image[image > (HU_tissue_range[1]-HU_tissue_range[0]) ] = (HU_tissue_range[1]-HU_tissue_range[0])
    image[image<0] = 0
    return image.astype(np.int16)
def normalize(image, HU_tissue_range):
    # tissue range [-1000, 400]
    image = (image - HU_tissue_range[0]) / float((HU_tissue_range[1]- HU_tissue_range[0]))
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    PIXEL_MEAN = 0.25 # value from LUNA16 data preprocessing tutorial
    image = image - PIXEL_MEAN
    return image
def get_lung_wings_crop_idx(pred, crop_coords, rescaling):
    _,thresh_pred = cv2.threshold(pred.copy(),128,255,cv2.THRESH_BINARY)
    thresh_pred = thresh_pred.astype(np.uint8)
    _, contours, _ = cv2.findContours(thresh_pred.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(thresh_pred.shape)== 2:
        thresh_pred = np.expand_dims(thresh_pred,2)
    min_y = []
    max_y = []
    min_x = []
    max_x = []
    # delete too small contours
    for cnt in contours:
        if cv2.contourArea(cnt)<5: continue
        if cnt.shape[0]<3: continue
        min_y.append(np.min(cnt[:,0,1]))
        max_y.append(np.max(cnt[:,0,1]))
        min_x.append(np.min(cnt[:,0,0]))
        max_x.append(np.max(cnt[:,0,0]))
    if bool(min_y) and bool(min_x) and bool(max_y) and bool(max_x):
        min_y = max(0,min(min_y))
        max_y = min(max(max_y), crop_coords[0])
        min_x = max(0,min(min_x))
        max_x = min(max(max_x), crop_coords[1])
        # scale back to original img_array.shape
        min_y = int(min_y*float(rescaling[0]))
        max_y = int(max_y*float(rescaling[0]))
        min_x = int(min_x*float(rescaling[1]))
        max_x = int(max_x*float(rescaling[1]))
        return [min_y, max_y, min_x, max_x]
    else:
        return False
def lung_wings_seg_preprocessing(img, config, scale, HU_tissue_range, data_type):
    if data_type == 'float32':
    # value range [-0.25,0.75] -> [0.0, 255.0]
        img += 0.25
        img *= 255.
        img = img.astype(np.uint8)
    elif data_type == 'int16':
        img = ((img / float((HU_tissue_range[1] - HU_tissue_range[0])))*255).astype(np.uint8)
    crop_coords = [int(x) for x in np.array(img.shape[:2])*scale]
    img_out = np.zeros((config['image_shape'][:2]+[img.shape[2]]), dtype=np.float32)
    for layer_cnt in range(img.shape[2]):
        img_out[:crop_coords[0],:crop_coords[1],layer_cnt] = cv2.resize(img[:,:,layer_cnt], tuple([crop_coords[1],crop_coords[0]]), interpolation=cv2.INTER_AREA)
        # cv2.imwrite('layer{}.jpg'.format(layer_cnt), img_out[:,:,layer_cnt])
    if not len(img_out.shape) == 3:
        print ('wrong shape')
        sys.exit()
    # [y,x,z] -> [z,y,x]
    img_out = np.rollaxis(img_out, 2,0)
    # expand with channel dimension
    img_out = np.expand_dims(img_out,3)
    img_out -= 128.0
    img_out /= 128.0
    return img_out, crop_coords
def lung_wings_seg_postprocessing(prediction):
    prediction *= 255.0
    prediction = prediction.astype(np.uint8)
    return prediction
def load_network(config, net, net_name, checkpoint_dir, out_dir, gpu_fraction):
    # Create a new eval directory, where to save predictions, all.log and config.json
    logging.info('Saving evaluation results to: {}'.format(out_dir))
    if not tf.gfile.Exists(out_dir):
        tf.gfile.MakeDirs(out_dir)
    initialize_logger(folder=out_dir)
    try:
        # Get model graph
        my_model_graph = net
    except:
        logging.error('(model_graph) was not provided!')
        raise KeyError('(model_graph) was not provided!')
    config['VARIABLES_TO_RESTORE'] = tf.contrib.slim.get_variables_to_restore()
    config['UPDATE_OPS_COLLECTION'] = tf.GraphKeys.UPDATE_OPS
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    init_op = tf.initialize_all_variables()
    # Define a session
    sess = tf.Session(config=tf.ConfigProto(
                      gpu_options = gpu_options,
                      allow_soft_placement=True,
                      log_device_placement=config['allow_soft_placement']))
    # Initialize all variables
    sess.run(init_op)
    data={}
    data['images'] = tf.placeholder(dtype=tf.float32, shape=[None] + config['image_shape'], name='image_placeholder')
    with tf.device('/gpu:' + ','.join([str(i) for i in H['lung_wings_seg_GPUs']])):
        config['dropout'] = 1.0
        out_, endpoints = my_model_graph(data=data,
                restore_logits=False,
                is_training=False,
                reuse=None,
                scope=net_name,
                **config)
    # Add all endpoints to predict
    config['endpoints'] = ['probs', 'logits']
    pred_ops = {}
    for key in config['endpoints']:
        if key in endpoints and key not in pred_ops:
            pred_ops[key] = endpoints[key]
        elif key in out_ and key not in pred_ops:
            pred_ops[key] = out_[key]
    if len(pred_ops) != len(config['endpoints']):
        logging.error('Not all enpoints found in the graph!: {}'.format(config['endpoints']))
        raise ValueError('Not all enpoints found in the graph!: {}'.format(config['endpoints']))
    # restore checkpoint and create saver
    if checkpoint_dir:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        var_lst = []
        for var in tf.all_variables():
            if var.name.split('/')[0] == net_name:
                var_lst.append(var)
        saver = tf_saver.Saver(var_lst, write_version=tf.train.SaverDef.V2)
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        logging.info('Succesfully loaded model from %s' % (checkpoint_dir))
    else:
        raise IOError('ckpt not found: {}'.format(checkpoint_dir))
        sys.exit()
    return sess, pred_ops, data
def main(test_pipeline):

    lung_wings_gpu_fraction = H['lung_wings_seg_gpu_fraction']
    if lung_wings_gpu_fraction > 0.88:
        lung_wings_gpu_fraction = 0.88
        print ('MAX gpu fraction is 0.88 due to mystical interactions with joblibs')

    #################################################################
    dataset_name = H['dataset_name']
    dataset_dir = H['dataset_dir']
    target_spacing = H['target_spacing'] # y,x,z
    target_spacing_zyx = np.array([target_spacing[2],target_spacing[0],target_spacing[1]]) # z,y,x
    HU_tissue_range = H['HU_tissue_range']
    data_type = H['resampled_lung_data_type']
    batch_size = H['lung_wings_seg_batch_size']
    num_threads = H['gen_resampled_lung_num_threads']
    if not bool(num_threads):
        num_threads = multiprocessing.cpu_count()-2
    if num_threads>multiprocessing.cpu_count()-1:
        print ('NOT all cpus should be used. Therefore reduced to multiprocessing.cpu_count()-1')
        num_threads=multiprocessing.cpu_count()-1
    print ('Using {} threads'.format(num_threads))
    lung_bounding_box_buffer_xy_px = H['lung_bounding_box_buffer_xy_px']   # y,x
    #################################################################
    ensure_dir(dataset_dir+'resampled_lungs/')
    # create dataset_json including all paths, informations,...
    dataset_json = {}
    dataset_json['dataset_name'] = dataset_name
    dataset_json['dataset_dir'] = dataset_dir
    dataset_json['lung_bounding_box_buffer_xy_px'] = lung_bounding_box_buffer_xy_px
    dataset_json['data_type'] = data_type
    dataset_json['HU_tissue_range'] = HU_tissue_range
    # load lung_wings segmentation
    lung_wings_config = json.load(open(lung_wings_seg_checkpoint_dir+'/config.json'))
    lung_wings_label_shape = lung_wings_config['label_shape']
    lung_wings_sess, lung_wings_pred_ops, lung_wings_data = load_network(lung_wings_config, lung_wings_unet_model.unet, lung_wings_config['model_name'], lung_wings_seg_checkpoint_dir, dataset_dir, lung_wings_gpu_fraction)
    dataset_json['lung_wings_segmentation_model_name'] = lung_wings_config['model_name']
    # load data
    if 1:
        # loop over subsets to generate list of all patients
        if H['data_2_process'] == 'LUNA16':
            print ('LUNA16')
            in_dir = H['LUNA16_raw_data_in_dir']
            subsets_patients_paths = glob(in_dir+"subset*/*.mhd")
            dataset_json['all_patients_paths']=subsets_patients_paths
            file_type = '.mhd'
            dataset_json['all_patients_paths'] = natsorted(dataset_json['all_patients_paths'], key=lambda p: p.split('/')[-1])
            dataset_json['all_patients'] = [patient.split('/')[-1].split('.')[-2] for patient in dataset_json['all_patients_paths']]
        elif H['data_2_process'] == 'dsb3':
            print ('dsb3')
            in_dir = H['dsb3_raw_data_in_dir']
            patients_paths = glob(in_dir+'*/')
            dataset_json['all_patients_paths']=patients_paths
            file_type = '.dcom'
            dataset_json['all_patients_paths'] = natsorted(dataset_json['all_patients_paths'], key=lambda p: p.split('/')[-1])
            dataset_json['all_patients'] = [patient.split('/')[-2] for patient in dataset_json['all_patients_paths']]
    if test_pipeline:
        dataset_json['all_patients'] = dataset_json['all_patients'][:test_pipeline]
        dataset_json['all_patients_paths'] = dataset_json['all_patients_paths'][:test_pipeline]
    dataset_json['patients'] = {}
    # list of junks with patients_lst within junks
    patients_junks = []
    num_junks = int(np.ceil(len(list(dataset_json['all_patients'])) / num_threads ) )
    for junk_cnt in range(num_junks):
        junk = []
        for in_junk_cnt in range(num_threads):
            patient_cnt = num_threads*junk_cnt + in_junk_cnt
            # break after last patient
            if patient_cnt >= len(dataset_json['all_patients']): break
            patient = dataset_json['all_patients'][patient_cnt]
            dataset_json['patients'][patient] = {}
            patient_json = dataset_json['patients'][patient]
            patient_json['patients_name'] = patient
            patient_json['raw_scan_path'] = dataset_json['all_patients_paths'][patient_cnt]
            junk.append(patient_json)
        patients_junks.append(junk)
    for junk_cnt, junk in enumerate(tqdm(patients_junks)):
        start_junk = time.time()
        img_array_junk = []
        for pa_json in junk:
            if file_type == '.mhd':
                img_array, spacing, origin, acquisition_exception_json = get_img_array_mhd(pa_json['raw_scan_path'])
            elif file_type == '.dcom':
                img_array, spacing, origin, acquisition_exception_json = get_img_array_dcom(pa_json['raw_scan_path'])
            else:
                print ('file type {} not supported!!!. Choose .dcom or .mhd.')
                sys.exit()
            original_shape = img_array.shape
            img_array_junk.append([img_array, spacing, origin, original_shape, acquisition_exception_json, pa_json['patients_name']]) # z,y,x
        #--------------------------------multithread-------------------------------------
        # heterogenous spacing -> homogeneous spacing
        resampled_junk_lst = Parallel(n_jobs=min([num_threads,len(junk)]))(delayed(resample)(pa_lst, target_spacing_zyx, data_type) for pa_lst in img_array_junk)
        #-------------------------------multithread-------------------------------------
        for pa_cnt, pa_lst in enumerate(resampled_junk_lst):
            # get results from multithreaded process
            img_array = pa_lst[0]
            spacing = pa_lst[1]
            real_spacing = pa_lst[2]
            origin = pa_lst[3]
            original_shape = pa_lst[4]
            acquisition_exception_json = pa_lst[5]
            patient = pa_lst[6]
            patient_json = dataset_json['patients'][patient]
            # resample img_array to homogeneous mm scale instead od px scale
            img_array = np.swapaxes(np.rollaxis(img_array,0,2),1,2) # y,x,z
            height, width, num_layers = img_array.shape
            spacing = [float(spacing[1]),float(spacing[2]),float(spacing[0])] # y,x,z
            real_spacing = [float(real_spacing[1]),float(real_spacing[2]),float(real_spacing[0])] # y,x,z
            pre_norm_value_hist, value_range = get_pre_normed_value_hist(img_array)
            if data_type == 'float32':
                img_array = zero_center(normalize(img_array,HU_tissue_range))
            elif data_type == 'int16':
                img_array = clip_HU_range(img_array, HU_tissue_range)
            else:
                print ('invalid data_type. use int16 or float32')
                sys.exit()
            # for layer_cnt in range(img_array.shape[2]):
            #     layer = img_array[:,:,layer_cnt,:]
            #     cv2.imshow('layer', ((layer+0.25)*255).astype(np.uint8))
            #     cv2.waitKey(0)
            # for some statistics
            patient_json['raw_scan_shape_yxz_px'] = original_shape
            patient_json['acquisition_exception'] = acquisition_exception_json
            patient_json['origin_yxz_mm'] = origin
            patient_json['target_spacing_yxz_mm/px'] = target_spacing
            patient_json['resampled_scan_shape_yxz_px'] = [height,width,num_layers]
            patient_json['raw_scan_spacing_yxz_mm/px'] = spacing
            patient_json['resampled_scan_spacing_yxz_mm/px'] = real_spacing
            patient_json['pre_normalized_zero-centered_value_histogram'] = [float(x) for x in pre_norm_value_hist[:]]
            patient_json['pre_normalized_zero-centered_value_range'] = [float(x) for x in value_range[:]]
            # lung wings segmentation # 512 due to embedding_shape of lungwings_segmentation training_data
            lung_wings_seg_max_shape = [int(H['lung_wings_seg_max_shape_trained_on'][0]/real_spacing[0]),int(H['lung_wings_seg_max_shape_trained_on'][1]/real_spacing[1])]
            # value range [-1.0, 1.0] axis [z,y,x,1]
            scale = [float(x) for x in np.array(lung_wings_config['image_shape'][:2])/lung_wings_seg_max_shape[:2]]
            img_array_for_lung_wings_seg, lung_wings_crop_coords = lung_wings_seg_preprocessing(img_array.copy(), lung_wings_config, scale, HU_tissue_range,  data_type)
            # calculate resacling factor for whole scan
            rescaling = [1.0/s for s in scale] # y,x
            # define lung_wings_crop_coords
            patient_json['lung_wings_crop_coords_yxz_px'] = {}
            patient_crop_coords = patient_json['lung_wings_crop_coords_yxz_px']
            patient_json['lung_wings_crop_shapes_yxz_px'] = {}
            ################## lung wings_segmentation
            num_batches = int(np.ceil(img_array.shape[2] / batch_size))
            for batch_cnt in range(num_batches):
                batch = np.ones([batch_size]+lung_wings_config['image_shape'][:3], dtype=np.float32)*(-1)
                z_crop_idx = [batch_cnt*batch_size,min((batch_cnt+1)*batch_size, img_array_for_lung_wings_seg.shape[0])]
                batch[:z_crop_idx[1]-z_crop_idx[0],:,:,:] = img_array_for_lung_wings_seg[z_crop_idx[0]:z_crop_idx[1],:,:,:]
                # lung_wings segmentation
                prediction = lung_wings_sess.run(lung_wings_pred_ops, feed_dict = {lung_wings_data['images']: batch})['probs']
                prediction = np.reshape(prediction, tuple([batch_size]+lung_wings_label_shape[:2]+[1]))
                prediction = lung_wings_seg_postprocessing(prediction)
                # evaluate prediction -> get lung_wings_crop idx
                for layer_in_batch_cnt in range(z_crop_idx[1]-z_crop_idx[0]):
                    layer_cnt = layer_in_batch_cnt+batch_size*batch_cnt
                    layer_crop_coords = get_lung_wings_crop_idx(prediction[layer_in_batch_cnt, :, :, :], lung_wings_crop_coords, rescaling)
                    if layer_crop_coords:
                        crop_shape = ( int(layer_crop_coords[1]-layer_crop_coords[0]), int(layer_crop_coords[3]-layer_crop_coords[2]), 1)
                        patient_json['lung_wings_crop_shapes_yxz_px'][layer_cnt] = crop_shape
                        patient_crop_coords[layer_cnt] = layer_crop_coords
            # crop bounding_cube around lung wings and save
            layers_coords = [ [patient_crop_coords[layer_cnt][x] for layer_cnt in patient_crop_coords.keys()] for x in range(4) ]
            if [True, True, True, True] == [True if len(x)>0 else False for x in layers_coords]:
                lung_coords = [max(0,min(layers_coords[0])-lung_bounding_box_buffer_xy_px[0]),
			      min(img_array.shape[0],max(layers_coords[1])+lung_bounding_box_buffer_xy_px[0]),
			      max(0,min(layers_coords[2])-lung_bounding_box_buffer_xy_px[1]),
			      min(img_array.shape[1],max(layers_coords[3])+lung_bounding_box_buffer_xy_px[1])]
            else:
                print ('WARNING - No lung wings found in scan of patient {}'.format(patient))
                print ('taking whole scan.')
                lung_coords = [0, img_array.shape[0], 0, img_array.shape[1]]
            patient_json['lung_bounding_cube_coords_yx_px'] = lung_coords
            # save .npy-float or .npy-int16 constructs
            patient_json['resampled_lung_path'] = dataset_dir+'resampled_lungs/'+patient+'_img.npy'
            print (patient_json['resampled_lung_path'])
            resampled_lung = img_array[lung_coords[0]:lung_coords[1], lung_coords[2]:lung_coords[3],:].copy()
            del(img_array)
            np.save(patient_json['resampled_lung_path'], resampled_lung)
            print ('patients processed {} / {}'.format(junk_cnt*num_threads+pa_cnt+1, len(dataset_json['all_patients'])))
        print ('time per patient {}'.format((time.time()-start_junk)/float(len(resampled_junk_lst))))
    # save dataset_json
    json.dump(dataset_json, open(dataset_dir + dataset_name+'_resample_lungs.json', 'w'), indent=4)
    print(dataset_dir + dataset_name+'_resample_lungs.json')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description="process dataset")
    parser.add_argument('--test_pipeline', type=int, required=False, default=False, help="choose num of patients processed")
    args = parser.parse_args()
    if args.test_pipeline:
        test_pipeline = args.test_pipeline
    elif H['test_pipeline']:
        test_pipeline = H['test_pipeline']
    else:
        test_pipeline = False
    if H['lung_wings_seg_GPUs'] != None: os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in H['lung_wings_seg_GPUs']])
    np.random.seed(17) # do NOT change
    main(test_pipeline)

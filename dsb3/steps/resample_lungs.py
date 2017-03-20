import numpy as np
import os, sys
import cv2
import SimpleITK as sitk
import scipy.ndimage
import dicom
import json
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import OrderedDict
from .. import utils
from .. import tf_tools
from .. import pipeline as pipe

logger = None # use this for logging

def run(target_spacing_zyx,
        bounding_box_buffer_yx_px,
        data_type,
        HU_tissue_range,
        checkpoint_dir,
        batch_size,
        seg_max_shape_yx):
    """
    Parameters
    ----------
    target_spacing_zyx : np.ndarray
        Target spacing of interpolated lung.
    bounding_box_buffer_yx_px : np.ndarray
        Target spacing of interpolated lung.
    data_type : {float32, int16}
        Output data type.
    HU_tissue_range : list of int
        [-1000, 400]
    checkpoint_dir : str
        Checkpoint directory of lung wings segmentation.
    batch_size : int
        Batch size for lung wings segmentation.
    seg_max_shape_yx : list of int
        [512, 512]
    """
    global logger
    logger = utils.get_logger(pipe.step_dir + 'step.log')
    if data_type not in ['float32', 'int16']:
        raise ValueError('Invalid data_type. Use int16 or float32.')
    if not os.path.exists(checkpoint_dir):
        raise ValueError('checkpoint_dir ' + checkpoint_dir + ' does not exist.')
    # heterogenous spacing -> homogeneous spacing
    print('... resizing and interpolating scans')
    patients_dict = dict(Parallel(n_jobs=min(pipe.n_CPUs, len(pipe.patients)), verbose=100)(
                                  delayed(resample_scan)(patient, target_spacing_zyx, data_type) 
                                  for patient in pipe.patients_raw_data_paths))
    print('... segmenting lung wings and cropping the scan')
    for patient, patient_dict in tqdm(patients_dict.items()):
        config = json.load(open(checkpoint_dir + '/config.json'))
        img_array_zyx = patient_dict['img_array_zyx']; del patient_dict['img_array_zyx']
        pre_norm_value_hist, value_range = get_pre_normed_value_hist(img_array_zyx)
        patient_dict['pre_normalized_zero-centered_value_histogram'] = [x for x in pre_norm_value_hist]
        patient_dict['pre_normalized_zero-centered_value_range'] = [x for x in value_range]
        img_array_zyx = clip_HU_range(img_array_zyx, HU_tissue_range)
        # lung wings segmentation (max width 512 due to embedding_shape of lungwings_segmentation training_data)
        seg_max_shape_yx = [int(seg_max_shape_yx[0] / patient_dict['resampled_scan_spacing_zyx_mm'][1]), 
                            int(seg_max_shape_yx[1] / patient_dict['resampled_scan_spacing_zyx_mm'][2])]
        # value range [-1.0, 1.0] axis [z, y, x, 1]
        scale_yx = [x for x in np.array(config['image_shape'][:2]) / seg_max_shape_yx[:2]] # config['image_shape'] is y, x
        img_array_seg_zyx, crop_coords_seg_yx = seg_preprocessing(img_array_zyx, config, scale_yx, HU_tissue_range)
        # calculate rescaling factor for whole scan
        inverse_scale_yx = [1.0/s for s in scale_yx] # y, x
        # define crop_coords_seg_yx
        patient_dict['crop_coords_zyx_px'] = []
        patient_dict['crop_shapes_zyx_px'] = []
        # lung_wings segmentation
        sess, pred_ops, data = tf_tools.load_network(checkpoint_dir)
        n_batches = int(np.ceil(img_array_zyx.shape[0] / batch_size))
        for batch_cnt in range(n_batches):
            batch = (-1) * np.ones([batch_size] + config['image_shape'], dtype=np.float32)
            z_crop_idx = [batch_cnt * batch_size, min((batch_cnt + 1) * batch_size, img_array_seg_zyx.shape[0])]
            batch[:z_crop_idx[1] - z_crop_idx[0], :, :, :] = img_array_seg_zyx[z_crop_idx[0] : z_crop_idx[1], :, :, :]
            # lung_wings segmentation
            prediction = sess.run(pred_ops, feed_dict = {data['images']: batch})['probs']
            prediction = np.reshape(prediction, tuple([batch_size] + config['label_shape'][:2] + [1]))
            prediction = seg_postprocessing(prediction)
            # evaluate prediction -> get crop idx
            for layer_in_batch_cnt in range(z_crop_idx[1] - z_crop_idx[0]):
                layer_cnt = layer_in_batch_cnt + batch_size * batch_cnt
                crop_coords = get_crop_idx(prediction[layer_in_batch_cnt, :, :, :], crop_coords_seg_yx, inverse_scale_yx)
                if crop_coords:
                    crop_shape = (int(crop_coords[1] - crop_coords[0]), int(crop_coords[3] - crop_coords[2]), 1)
                    patient_dict['crop_shapes_zyx_px'] += [crop_shape]
                    patient_dict['crop_coords_zyx_px'] += [crop_coords]
        # crop bounding_cube around lung wings and save
        layers_coords = [[coords[x] for coords in patient_dict['crop_coords_zyx_px']] for x in range(4)]
        if [True, True, True, True] == [True if len(x) > 0 else False for x in layers_coords]:
            lung_coords = [max(0, min(layers_coords[0]) - bounding_box_buffer_yx_px[0]),
                           min(img_array_zyx.shape[0], max(layers_coords[1]) + bounding_box_buffer_yx_px[0]),
                           max(0, min(layers_coords[2]) - bounding_box_buffer_yx_px[1]),
                           min(img_array_zyx.shape[1], max(layers_coords[3]) + bounding_box_buffer_yx_px[1])]
        else:
            logger.warning('No lung wings found in scan of patient  {}'.format(patient) + '. Taking the whole scan.')
            lung_coords = [0, img_array_zyx.shape[0], 0, img_array_zyx.shape[1]]
        patient_dict['bounding_box_coords_yx_px'] = lung_coords
        patient_dict['resampled_lung_path'] = pipe.step_dir_data + patient + '_img.npy'
        patients_dict[patient] = patient_dict
        np.save(patient_dict['resampled_lung_path'], img_array_zyx[:, lung_coords[0]:lung_coords[1], lung_coords[2]:lung_coords[3]])
    sess.close()
    results_dict = OrderedDict([('HU_tissue_range', HU_tissue_range)])
    results_dict.update(patients_dict)
    pipe.save_step(results_dict)

def get_pre_normed_value_hist(img_array):
    hist,ran = np.histogram(img_array.flatten(), bins=16*5,normed=True, range=[-1000,600])
    return hist, ran

def get_img_array_mhd(img_file):
    itk_img = sitk.ReadImage(img_file)
    img_array_zyx = sitk.GetArrayFromImage(itk_img) # indices are z, y, x 
    origin = itk_img.GetOrigin() # x, y, z  world coordinates (mm)
    origin_zyx = [origin[2], origin[1], origin[0]] # y, x, z
    spacing = itk_img.GetSpacing() # x, y, z world coordinates (mm)
    spacing_zyx = [spacing[2], spacing[1], spacing[0]] # z, y, x
    acquisition_exception = None # no acquisition number found in object
    return img_array_zyx, spacing_zyx, origin_zyx, acquisition_exception

def get_img_array_dcom(img_file):
    def load_scan(path):
        patient = path.split('/')[-2]
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        unique_ac_nums, counts = np.unique([s.AcquisitionNumber for s in slices], return_counts = True)
        if len(unique_ac_nums) > 1:
            counts = [int(i) for i in counts]
            logger.warning('Multiple scan exception, different acquisition numbers: {}'.format(unique_ac_nums))
            logger.warning('    Counts: {}'.format(counts))
            logger.warning('    Patient {}'.format(patient))
            # celecting the index with the highest number of acquisitions. in case of balanced acquisitions,
            # selecting the latter, operation is string compatible
            selected_acquisition = unique_ac_nums[np.argwhere(counts == np.amax(counts))[-1][0]]
            logger.warning('proceding with most frequent acquisition number {}'.format(selected_acquisition))
            slices = [s for s in slices if s.AcquisitionNumber == selected_acquisition]
            acquisition_exception = {}
            multiple_scan_exception = [[str(i) for i in unique_ac_nums], counts]
            acquisition_exception['multiple_scan_exception_uniques/counts'] = multiple_scan_exception
            acquisition_exception['selected_acquisition'] = str(selected_acquisition)
        elif len(unique_ac_nums) == 0:
            acquisition_exception = 'No AcquisitionNumber'
            logger.warning('Patient {} without acquisition number.'.format(patient))
        else:
            acquisition_exception = None
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        spacings = [s.PixelSpacing for s in slices]
        for s in slices:
            s.SliceThickness = slice_thickness            
            for i in range(2): # check spacing
                if math.isnan(s.PixelSpacing[i]) or s.PixelSpacing[i] < 0.1 or s.PixelSpacing[i] > 5 or isinstance(s.PixelSpacing[i], str): # check for exceptions
                    # clean list of exceptions before taking mode.
                    cleaned_spacings = [sp[i] for sp in spacings if not math.isnan(sp[i]) if not sp[i] < 0.1 if not sp[i] > 5 if not isinstance(sp[i], str)]
                    # take the mode value of spacings, or the first value if values are even
                    s.PixelSpacing[i] = np.argmax(np.bincount(cleaned_spacings))
        return slices, acquisition_exception
    def get_pixels_hu(slices):
        image = np.stack([s.pixel_array for s in slices])
        # convert to int16 (from sometimes int16) should be possible as values should always be low enough (<32k).
        if np.max(image) > np.iinfo(np.int16).max:
            logger.error('Controlled ransformation of pixel array to np.int16 failed: too high values!')
        image = image.astype(np.int16)
        # convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)
        return np.array(image, dtype=np.int16)
    scan, acquisition_exception = load_scan(img_file)
    img_array_zyx = get_pixels_hu(scan) # z, y, x
    spacing_zyx = np.array(list(map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing)))) # z, y, x
    origin_zyx = None
    return img_array_zyx, spacing_zyx, origin_zyx, acquisition_exception

def resample_scan(patient, target_spacing_zyx, data_type):
    if pipe.dataset_name == 'LUNA16':
        img_array_zyx, old_spacing_zyx, origin_zyx, acquisition_exception = get_img_array_mhd(pipe.patients_raw_data_paths[patient])
    elif pipe.dataset_name == 'dsb3':
        img_array_zyx, old_spacing_zyx, origin_zyx, acquisition_exception = get_img_array_dcom(pipe.patients_raw_data_paths[patient])
    new_shape_zyx = np.round(img_array_zyx.shape * np.array(old_spacing_zyx) / np.array(target_spacing_zyx))
    resize_factor = new_shape_zyx / img_array_zyx.shape
    new_spacing_zyx = list(np.array(old_spacing_zyx) / np.array(resize_factor))
    old_shape_zyx_px = img_array_zyx.shape
    img_array_zyx = scipy.ndimage.interpolation.zoom(img_array_zyx if data_type=='int16' else img_array.astype(np.float32), resize_factor, order=1, mode='nearest')
    return patient, {'img_array_zyx': img_array_zyx,
                     'resampled_scan_spacing_zyx_mm': new_spacing_zyx,
                     'resampled_scan_shape_zyx_px': img_array_zyx.shape,
                     'raw_scan_spacing_zyx_mm': old_spacing_zyx,
                     'raw_scan_shape_zyx_px': old_shape_zyx_px,
                     'raw_scan_origin_zyx_mm': origin_zyx,
                     'acquisition_exception': acquisition_exception}

def clip_HU_range(img_array, HU_tissue_range):
    if img_array.dtype == np.int16:
        return clip_HU_range_int16(img_array, HU_tissue_range)
    elif img_array.dtype == np.float32:
        return zero_center(normalize_HU_range_float(img_array, HU_tissue_range))
    else:
        raise ValueError('Array has wrong data type.')

def clip_HU_range_int16(img_array, HU_tissue_range):
    img_array = img_array - HU_tissue_range[0] # tissue range [-1000, 400]
    img_array[img_array > (HU_tissue_range[1] - HU_tissue_range[0])] = HU_tissue_range[1] - HU_tissue_range[0]
    img_array[img_array < 0] = 0
    return img_array.astype(np.int16)

def normalize_HU_range_float(img_array, HU_tissue_range):
    img_array = (img_array - HU_tissue_range[0]) / float((HU_tissue_range[1]- HU_tissue_range[0])) # tissue range [-1000, 400]
    img_array[img_array > 1] = 1.
    img_array[img_array < 0] = 0.
    return img_array

def zero_center(img_array):
    pixel_mean = 0.25 # value from LUNA16 data preprocessing tutorial
    img_array = img_array - pixel_mean
    return img_array

def get_crop_idx(pred, crop_coords, invers_scale_yx):
    _, thresh_pred = cv2.threshold(pred.copy(), 128, 255, cv2.THRESH_BINARY)
    thresh_pred = thresh_pred.astype(np.uint8)
    _, contours, _ = cv2.findContours(thresh_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(thresh_pred.shape) == 2:
        thresh_pred = np.expand_dims(thresh_pred,2)
    min_y = []
    max_y = []
    min_x = []
    max_x = []
    # delete too small contours
    for cnt in contours:
        if cv2.contourArea(cnt) < 5: continue
        if cnt.shape[0] < 3: continue
        min_y.append(np.min(cnt[:, 0, 1]))
        max_y.append(np.max(cnt[:, 0, 1]))
        min_x.append(np.min(cnt[:, 0, 0]))
        max_x.append(np.max(cnt[:, 0, 0]))
    if min_y and min_x and max_y and max_x:
        min_y = max(0, min(min_y))
        max_y = min(max(max_y), crop_coords[0])
        min_x = max(0, min(min_x))
        max_x = min(max(max_x), crop_coords[1])
        # scale back to original img_array.shape
        min_y = int(min_y * float(invers_scale_yx[0]))
        max_y = int(max_y * float(invers_scale_yx[0]))
        min_x = int(min_x * float(invers_scale_yx[1]))
        max_x = int(max_x * float(invers_scale_yx[1]))
        return [min_y, max_y, min_x, max_x]
    else:
        return []

def seg_preprocessing(img_array_zyx, config, scale_yx, HU_tissue_range):
    # transform to np.unit8 to apply cv2.imwrite
    if img_array_zyx.dtype == np.float32: # value range [-0.25, 0.75] -> [0, 255]
        img_array_zyx += 0.25
        img_array_zyx *= 255
        if np.max(img_array_zyx >= 256.):
            logger.error('Data transformation did not work! Observed values greater than 256 before transforming to uint8!')
        img_array_zyx = img_array_zyx.astype(np.uint8)
    elif img_array_zyx.dtype == np.int16:
        img_array_zyx = ((img_array_zyx / (HU_tissue_range[1] - HU_tissue_range[0])) * 255).astype(np.uint8)
    crop_coords_yx = [int(x) for x in np.array(img_array_zyx.shape[1:]) * scale_yx]
    img_array_zyx_out = np.zeros(([img_array_zyx.shape[0]] + config['image_shape'][:2]), dtype=np.float32) # config['image_shape'] is y, x, z
    for layer_cnt in range(img_array_zyx.shape[0]):
        img_array_zyx_out[layer_cnt, :crop_coords_yx[0], :crop_coords_yx[1]] = cv2.resize(img_array_zyx[layer_cnt, :, :],
                                                                                          tuple(crop_coords_yx),
                                                                                          interpolation=cv2.INTER_AREA)
        # cv2.imwrite('layer{}.jpg'.format(layer_cnt), img_array_zyx_out[layer_cnt, :, :])
    if not len(img_array_zyx_out.shape) == 3:
        logger.error('wrong shape of img_array_zyx_out in seg_preprocessing.')
    img_array_zyx_out = np.expand_dims(img_array_zyx_out, 3).astype(np.float32) # expand with channel dimension
    img_array_zyx_out -= 128
    img_array_zyx_out /= 128
    return img_array_zyx_out, crop_coords_yx

def seg_postprocessing(prediction):
    prediction *= 255
    return prediction.astype(np.uint8)

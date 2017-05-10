'''
Feed Iterators responsible for augmentation and batching
in pure python
'''
import sys, os
import logging
import cv2
import numpy as np
import pandas as pd
#from hellsicht.tensorflow.modules.config import conf
from io_modules.iterator import Iterator
import tensorflow as tf
import time
from tqdm import tqdm

import multiprocessing
from joblib import Parallel, delayed

np.random.seed(1713)

def rotate(in_tensor, M):
    dst = cv2.warpAffine(in_tensor,M,(in_tensor.shape[1],in_tensor.shape[0]), flags=cv2.INTER_CUBIC)
    return dst

def rotate_channels(tensor, M):
    for i in range(tensor.shape[2]):
        tensor[:,:,i] = rotate(tensor[:,:,i], M)
    return tensor

class NPY_construct_Iter(Iterator):
    def __init__(self, H, **conf):
        super(NPY_construct_Iter, self).__init__()

        # Read additional arguments from kwargs
        self.img_lst_path = conf['img_lst']
        self.npy_path = conf['npy_lst']
        self.in_img_shape = H['in_image_shape']
        self.img_shape = conf['img_shape']
        self.label_shape = conf['label_shape']
        self.batch_size = conf['batch_size']
        self.shuffle = conf['shuffle']
        self.is_training = conf['is_training']
        self.rand_crop=H['rand_crop']
        self.crop_min = H['crop_min']
        self.crop_max = H['crop_max']
        self.min_prio = H['min_prio']

        self.num_nodule_free_per_batch= H['num_nodule_free_per_batch']
        
        self.staged_intensities = H['staged_intensities']
        self.compensation = H['compensation']
        # self.crop_before_loading_in_RAM = H['crop_before_loading_in_RAM_ZminZmaxYminYmaxXminXmax']
        #self.crop_before_loading_in_RAM =
        #    self.crop_before_loading_in_RAM= [0, self.in_img_shape[2]]

        # Read augmentation parameters from hyperparameter file
        # self.rand_cropping_coords = H['rand_cropping_ZminZmaxYminYmaxXminXmax']
        # self.rand_cropping_coords = [0 if not v and cnt%2==0 else
        #                              self.in_img_shape[cnt//2+1] if not v and cnt%2==1 else int(v)
        #                              for cnt,v in enumerate(self.rand_cropping_coords)]

        self.rand_rot          = H['rand_rot']
        self.descreteRotAngles = H['descreteRotAngles']
        self.min_rot_angle     = H['min_rot_angle']
        self.max_rot_angle     = H['max_rot_angle']
        self.nintyDegRot       = H['nintyDegRot']

        # Declare placeholders for labels and images
        assert isinstance(self.label_shape, list)
        self.images = tf.placeholder(dtype=tf.float32, shape=[self.batch_size] + self.img_shape, name='image_placeholder')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size] + self.label_shape[:-1] + [3], name='label_placeholder')

        self.numpy_image_batch = None
        self.numpy_label_batch = None
        self.counter_nodule      = 0
        self.counter_nodule_free = 0
        self.index_list_nodule = []
        self.index_list_nodule_free = []

        self.unity_center_radius = H['unity_center_radius']

        #read mode
        self.load_in_ram = H['load_in_ram']
        self.load_lsts_fraction = H['load_lsts_fraction']

    def load_img_lst(self):
        try:
            self.img_lst = pd.read_csv(self.img_lst_path, sep='\t')
        except:
            raise IOError('File could not be opened!, {}'.format(self.img_lst_path))
        if self.load_lsts_fraction:
            self.img_lst = self.img_lst.reindex(np.random.permutation(self.img_lst.index))[0:int(len(self.img_lst)//self.load_lsts_fraction)]
            print ('using {} shuffled images from lst'.format(len(self.img_lst)))

        if self.load_in_ram:
            print("loading all images from: ", self.img_lst_path)
            self.data_pairs_nodule      = []
            self.data_pairs_nodule_free = []
            npy = np.load(self.npy_path)

            self.index_list_nodule = []
            nodule_cnt = 0

            for cnt, row in enumerate(tqdm(self.img_lst.iterrows())):

                img_cnt = int(row[1]['ongoing_num'])
                is_nodule = int(row[1]['is_nodule'])
                img = npy[img_cnt,:,:,:self.in_img_shape[-1],0].astype(np.uint8)
                labels = npy[img_cnt,:,:,0,1:3].astype(np.uint8)
                lab_mask2pred = labels[:,:,0]
                lab_center = labels[:,:,1]
                lab_center[lab_center<10] = 0 # delete noise

                
                # ensure min_prio
                if self.min_prio == 1:
                    pass
                elif self.min_prio == 2:
                    lab_mask2pred[lab_mask2pred<100] = 0
                    lab_center[lab_center<100] = 0
                elif self.min_prio == 3:
                    lab_mask2pred[lab_mask2pred<200] = 0
                    lab_center[lab_center<200] = 0
                # gen center mask with unity center size
                lab_center_in = lab_center.copy()


                # draw unity centers with same color(=nodule_priority) like in lab_center
                _, contours, _ = cv2.findContours(lab_center.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                lab_center_unity = np.zeros_like(lab_center, dtype=np.uint8)
                for contour in contours:
                    center = tuple([int(np.mean(contour[:,0,0])), int(np.mean(contour[:,0,1]))]) # xy
                    color  = int(lab_center[center[1], center[0]])
                    cv2.circle(lab_center_unity, center, self.unity_center_radius, (color), -1)
                lab_center = lab_center_unity

                if not self.staged_intensities:
                    lab_mask2pred[lab_mask2pred > 5] = 255

                # gen broaded mask
                lab_broaded = cv2.blur(lab_mask2pred.copy(), (7,7))
                
                lab_broaded[lab_broaded>1] = 255
                lab = cv2.merge([lab_mask2pred, lab_center, lab_broaded])

                if 0 and bool(is_nodule):
                    cv2.imwrite('test_imgs/'+str(cnt)+'.jpg',img)
                    cv2.imwrite('test_imgs/'+str(cnt)+'_lab.jpg',lab[:,:,0])
                    cv2.imwrite('test_imgs/'+str(cnt)+'_center.jpg',lab[:,:,1])
                    cv2.imwrite('test_imgs/'+str(cnt)+'_broaded.jpg',lab[:,:,2])
                    cv2.imwrite('test_imgs/'+str(cnt)+'_center_in.jpg',lab_center_in)
                    # if cnt == 975: break
                dict = {'data' : img,
                        'labels' : lab,
                        'is_nodule' : is_nodule}


                if is_nodule:
                    self.data_pairs_nodule.append(dict)
                    if self.is_training:
                        if self.compensation:
                            for i in range(int(round(row[1]['compensation_factor']))):
                                self.index_list_nodule.append(nodule_cnt)
                        else:
                            self.index_list_nodule.append(nodule_cnt)
                    else:
                        self.index_list_nodule.append(nodule_cnt)

                    nodule_cnt += 1
                else:
                    self.data_pairs_nodule_free.append(dict)

            #self.index_list_nodule = list(range(len(self.data_pairs_nodule)))
            self.index_list_nodule_free = list(range(len(self.data_pairs_nodule_free)))


            if self.is_training:
                self.shuffle_index_lst_nodule_()
                self.shuffle_index_lst_nodule_free_()


    def read_batch(self):
        # nodule part of batch
        batch_nodule_df      = self.index_list_nodule[self.counter_nodule * (self.batch_size-self.num_nodule_free_per_batch) : (self.counter_nodule + 1) * (self.batch_size-self.num_nodule_free_per_batch)]
        if len(batch_nodule_df) < self.batch_size-self.num_nodule_free_per_batch:
            batch_nodule_df_res = self.index_list_nodule[:(self.batch_size-self.num_nodule_free_per_batch - len(batch_nodule_df))]
            batch_nodule_df = batch_nodule_df + batch_nodule_df_res
            self.shuffle_index_lst_nodule_()
            self.counter_nodule = -1
        elif (self.counter_nodule + 1) * (self.batch_size-self.num_nodule_free_per_batch) == len(self.index_list_nodule):
            self.shuffle_index_lst_nodule_()
            self.counter_nodule = -1 

        self.numpy_image_batch = []
        self.numpy_label_batch = []

        for idx in batch_nodule_df:

            tup = self.data_pairs_nodule[idx]

            #load img
            data      = tup['data'].copy()
            labels    = tup['labels'].copy()
            id_nodule = tup['is_nodule']
            data, labels = self.AugmentData(data,labels)

            self.numpy_label_batch.append(labels)
            self.numpy_image_batch.append(data)

        # nodule_free part of batch
        batch_nodule_free_df = self.index_list_nodule_free[self.counter_nodule_free * (self.num_nodule_free_per_batch) : (self.counter_nodule_free + 1) * (self.num_nodule_free_per_batch)]
        if len(batch_nodule_free_df) < self.num_nodule_free_per_batch:
            batch_nodule_free_df_res = self.index_list_nodule_free[:(self.num_nodule_free_per_batch - len(batch_nodule_free_df))]
            batch_nodule_free_df = batch_nodule_free_df + batch_nodule_free_df_res
            self.shuffle_index_lst_nodule_free_()
            self.counter_nodule_free = -1
        elif (self.counter_nodule_free + 1) * self.num_nodule_free_per_batch == len(self.index_list_nodule_free):
            self.shuffle_index_lst_nodule_free_()
            self.counter_nodule_free = -1 

        for idx in batch_nodule_free_df:

            tup = self.data_pairs_nodule_free[idx]

            #load img
            data      = tup['data'].copy()
            labels    = tup['labels'].copy()
            id_nodule = tup['is_nodule']
            data, labels = self.AugmentData(data,labels)

            self.numpy_label_batch.append(labels)
            self.numpy_image_batch.append(data)


        self.numpy_image_batch = np.asarray(self.numpy_image_batch)
        self.numpy_label_batch = np.asarray(self.numpy_label_batch)

        # self.numpy_label_batch = np.reshape(self.numpy_label_batch, [self.batch_size, -1])

        self.counter_nodule      += 1
        self.counter_nodule_free += 1


    def shuffle_index_lst_nodule_(self):
        np.random.shuffle(self.index_list_nodule)
    def shuffle_index_lst_nodule_free_(self):
        np.random.shuffle(self.index_list_nodule_free)

    def shuffle_lst_(self):
        self.img_lst = self.img_lst.reindex(np.random.permutation(self.img_lst.index))

    def data_batch(self):
        return {'images' : self.images}

    def label_batch(self):
        return {'labels' : self.labels}

    def get_data_batch(self):
        return {'images' : self.numpy_image_batch}

    def get_label_batch(self):
        return {'labels' : self.numpy_label_batch}


    def rand_crop_data(self, data, labels, is_train):                
        
        #ToDO CROP from larger scan, dont pad with black

        #test
        if is_train:
            random_crop_size = np.random.randint(self.crop_min, self.crop_max)

        #Crop if smaller
        if random_crop_size < int(np.ceil((self.crop_max + self.crop_min) / 2.0)) :
            random_crop_start_x = np.random.randint(0, data.shape[0] - random_crop_size)
            random_crop_start_y = np.random.randint(0, data.shape[1] - random_crop_size)

            for layer in range(data.shape[2]):
                data[:,:,layer] = cv2.resize(data[random_crop_start_y:random_crop_start_y+random_crop_size,random_crop_start_x:random_crop_start_x+random_crop_size,layer], (data.shape[1], data.shape[0]))

            for layer in range(labels.shape[2]):
                labels[:,:,layer] = cv2.resize(labels[random_crop_start_y:random_crop_start_y+random_crop_size,random_crop_start_x:random_crop_start_x+random_crop_size,layer], (labels.shape[1], labels.shape[0]))
        elif random_crop_size > int(np.ceil((self.crop_max + self.crop_min) / 2.0)):
            frame_data = np.zeros(data.shape, dtype=np.uint8)
            frame_labels = np.zeros(labels.shape, dtype=np.uint8)
            for layer in range(data.shape[2]):     


                size_y, size_x = int((data.shape[1] ** 2) / random_crop_size), int((data.shape[0] ** 2) / random_crop_size)

                idy_start = (frame_data.shape[0] - size_y)//2
                idy_end = (frame_data.shape[0] - size_y)//2+size_y

                idx_start = (frame_data.shape[1] - size_x)//2
                idx_end = (frame_data.shape[1] - size_x)//2+size_x

                frame_data[idy_start:idy_end,idx_start:idx_end,layer] = cv2.resize(data[:,:,layer], (size_y, size_x )) 

            for layer in range(labels.shape[2]):                

                size_y, size_x = int((labels.shape[1] ** 2) / random_crop_size), int((labels.shape[0] ** 2) / random_crop_size)

                idy_start = (frame_labels.shape[0] - size_y)//2
                idy_end = (frame_labels.shape[0] - size_y)//2+size_y

                idx_start = (frame_labels.shape[1] - size_x)//2
                idx_end = (frame_labels.shape[1] - size_x)//2+size_x

                frame_labels[idy_start:idy_end,idx_start:idx_end,layer] = cv2.resize(labels[:,:,layer], (size_y, size_x )) 

            return frame_data, frame_labels
        return data, labels 

    def AugmentData(self, data, labels):

        if self.is_training:

            if self.rand_rot:

                if self.descreteRotAngles:
                    self.rand_angle_1 = np.random.permutation(self.descreteRotAngles)[0]
                else:
                    self.rand_angle_1 = np.random.randint(self.min_rot_angle, self.max_rot_angle)

                if self.nintyDegRot:
                    rot90= np.random.randint(0,4)
                    self.rand_angle_1 += rot90*90

                M = cv2.getRotationMatrix2D((data.shape[1]//2,data.shape[0]//2), self.rand_angle_1,1)

                data   = rotate_channels(data, M)
                labels = rotate_channels(labels, M)

            if self.rand_crop:
                data, labels = self.rand_crop_data(data, labels, self.is_training)


        #else:
        #    if self.rand_crop:
        #        data, labels = self.rand_crop_data(data, labels, self.is_training)

        # normalize data and labels and zero_center_data data
        ret_data = data.copy()/255.-0.25
        ret_labels = labels.copy()/255
        return ret_data, ret_labels








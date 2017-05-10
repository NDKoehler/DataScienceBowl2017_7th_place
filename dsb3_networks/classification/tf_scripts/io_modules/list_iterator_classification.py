'''
Feed Iterators responsible for augmentation and batching
in pure python
'''
import sys, os
import logging
import cv2
import numpy as np
import pandas as pd
# from hellsicht.tensorflow.modules.config import conf
from io_modules.iterator import Iterator
import tensorflow as tf
import time

import multiprocessing
from joblib import Parallel, delayed

np.random.seed(1713)

def rotate(in_tensor, M):
    dst = cv2.warpAffine(in_tensor, M, (in_tensor.shape[1], in_tensor.shape[0]), flags=cv2.INTER_CUBIC)
    if len(dst.shape) == 2:
        dst = np.expand_dims(dst, 2)
    return dst

def rotate_3d(tensor, M, rand_axis, rand_angle, crop_starts, img_shape, cross_axes=False):
    # rotate around z only in layers cropped later
    if rand_axis == 0:
        if not cross_axes:
            for i in np.arange(crop_starts[0], crop_starts[0]+img_shape[1]):
                tensor[i,:,:] = rotate(tensor[i,:,:], M)
        else:
            for i in range(tensor.shape[0]):
                tensor[i,:,:] = rotate(tensor[i,:,:], M)
    elif rand_axis == 1:
        for i in range(tensor.shape[1]):
            tensor[:, i,:] = rotate(tensor[:, i,:], M)
    elif rand_axis == 2:
        for i in range(tensor.shape[2]):
            tensor[:,:, i] = rotate(tensor[:,:, i], M)
    return tensor

def rotate_3d_multthread(tensor, M, rand_axis, rand_angle, crop_starts, img_shape):
    # rotate around z only in layers cropped later
    num_threads = 2#multiprocessing.cpu_count() - 2

    rand_axis = 1

    if rand_axis == 0:
        for i in np.arange(crop_starts[0], crop_starts[0]+img_shape[1]):
            tensor[i,:,:] = rotate(tensor[i,:,:], M)
    elif rand_axis == 1:
        # for i in range(tensor.shape[1]):
        #    tensor[:,i,:] = rotate(tensor[:,i,:], M)
        resampled_junk_lst = Parallel(n_jobs=num_threads)(delayed(rotate)(tensor[:, i,:].copy(), M) for i in range(tensor.shape[1]))      
        for i in range(tensor.shape[1]):
            tensor[:, i,:] = resampled_junk_lst[i]

    elif rand_axis == 2:
        for i in range(tensor.shape[2]):
            tensor[:,:, i] = rotate(tensor[:,:, i], M)
    return tensor

class List_Iterator(Iterator):
    def __init__(self, H, **conf):
        super(List_Iterator, self).__init__()


        # Read additional arguments from kwargs
        self.img_lst_path = conf['img_lst']
        self.in_img_shape = H['in_image_shape']
        self.img_shape = conf['img_shape']
        self.label_shape = conf['label_shape']
        self.batch_size = conf['batch_size']
        self.shuffle = conf['shuffle']
        self.is_training = conf['is_training']
        self.num_candidates = H['image_shape'][0]
        # self.crop_before_loading_in_RAM = H['crop_before_loading_in_RAM_ZminZmaxYminYmaxXminXmax']
        # self.crop_before_loading_in_RAM =
        #    self.crop_before_loading_in_RAM= [0, self.in_img_shape[2]]

        # Read augmentation parameters from hyperparameter file
        self.rand_cropping_coords = H['rand_cropping_ZminZmaxYminYmaxXminXmax']
        self.rand_cropping_coords = [0 if not v and cnt%2 == 0 else
                                     self.in_img_shape[cnt//2+1] if not v and cnt%2 == 1 else int(v)
                                     for cnt, v in enumerate(self.rand_cropping_coords)]

        self.rand_rot       = H['rand_rot']
        self.min_rot_angle  = H['min_rot_angle']
        self.max_rot_angle  = H['max_rot_angle']
        self.rand_mirror_axes = H['rand_mirror_axes']
        self.rand_rot_axes = H['rand_rot_axes']

        self.degree_90_rot = H['degree_90_rot']

        self.candidate_mode = H['candidate_mode']
        try:
            self.importance_sampling = H['importance_sampling'] * self.is_training
            self.importance_dict = H['importance_dict']
        except:
            self.importance_sampling = False
            self.importance_dict = None

        # crossed_axes mode
        self.cross_axes = H['crossed_axes']
        self.num_crossed_layers = H['num_crossed_layers']
        self.rand_drop_planes = H['rand_drop_planes']

        if not self.is_training:
            self.img_shape[1] = self.img_shape[1] + self.rand_drop_planes
            self.rand_drop_planes = False

        # Declare placeholders for labels and images
        assert isinstance(self.label_shape, list)

        self.images = tf.placeholder(dtype=tf.float32, shape=[self.batch_size] + self.img_shape, name='image_placeholder')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size] + self.label_shape, name='label_placeholder')
        self.tmp_cand_cropped = np.zeros(self.img_shape[1:], dtype = float)
        self.tmp_candidates   = np.zeros(self.img_shape, dtype = float)
        #else:
        #    self.images = tf.placeholder(dtype=tf.float32, shape=[self.batch_size] + self.in_img_shape[:1] + self.img_shape[1:], name='image_placeholder')
        #    self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size] + self.label_shape, name='label_placeholder')
        #    self.tmp_cand_cropped = np.zeros(self.in_img_shape[:1] + self.img_shape[1:], dtype = float)
        #    self.tmp_candidates   = np.zeros(self.in_img_shape[:1] + self.img_shape[1:], dtype = float)


        if self.cross_axes:
            if not self.img_shape[1] == len(self.cross_axes)*self.num_crossed_layers-self.rand_drop_planes:
                print ('WRONG img_shape for corssed axis mode. img_shape[1]==len(self.cross_axes)*self.num_crossed_layers)!!!')
                sys.exit()
            if not self.img_shape[2] == self.img_shape[3]:
                print ('WRONG img_shape. img_shape[2:] must be equal to img_shape[3:]!!!')
                sys.exit()
            self.tmp_crossed_layer = np.zeros(([len(self.cross_axes)*self.num_crossed_layers]+list(self.img_shape[2:])), dtype=np.float32)
            self.rand_cropping_coords = [0, self.in_img_shape[2], 0, self.in_img_shape[2], 0, self.in_img_shape[2]]
            self.tmp_cand         = np.zeros((self.img_shape[2], self.img_shape[2], self.img_shape[2], self.img_shape[-1]), dtype = np.uint8)
        else:
            self.tmp_cand         = np.zeros(self.in_img_shape[1:], dtype = np.uint8)


        self.numpy_image_batch = None
        self.numpy_label_batch = None
        self.counter = 0

        # read mode
        self.load_in_ram = H['load_in_ram']
    def load_data(self):
        try:
            self.img_lst = pd.read_csv(self.img_lst_path, header=None, sep='\t')
        except:
            raise IOError('File could not be opened!, {}'.format(self.img_lst_path))

        if self.load_in_ram:
            print("loading all images from: ", self.img_lst_path)
            self.data_pairs = []



            if not self.candidate_mode:

                for row in self.img_lst.iterrows():

                    img_path = row[1][2]
                    label = row[1][1]

                    # load img
                    if os.path.isfile(img_path):
                        img_raw = np.load(img_path).astype(np.uint8)
                        # img = img_raw[:self.in_img_shape[0], self.crop_before_loading_in_RAM[0]:self.crop_before_loading_in_RAM[1], :, :, :self.in_img_shape[4]].copy()
                        
                        z, y, x = self.in_img_shape[1:4]
                        z_raw, y_ray, x_ray = img_raw.shape[1:4]
                        diff = np.abs(z_raw - z)//2

                        if diff != 0:
                            img = img_raw[:self.in_img_shape[0], diff:-diff, diff:-diff, diff:-diff, :self.in_img_shape[4]].copy()
                        else:
                            img = img_raw[:self.in_img_shape[0],:,:,:, :self.in_img_shape[4]].copy()
                      
                        del(img_raw)
                        # img = np.expand_dims(img, 4)

                    else:
                        logging.error('Could not read image: {}'.format(img_path))
                        raise IOError('Could not read image: {}'.format(img_path))

                    dict = {'data' : img,
                            'labels' : label}

                    self.data_pairs.append(dict)
                    if(len(self.data_pairs) % 1000 == 0):
                        print(len(self.data_pairs), "data records in RAM")
                    #if len(self.data_pairs) % 100 == 0:
                    #    break


                self.index_list = list(range(len(self.data_pairs)))


            # candidate Mode
            if self.candidate_mode:

                self.index_list = []
                index_counter = 0

                all_patient_names = self.img_lst[0].values.tolist()
                all_patient_names = list(set([x.split('/')[-1].split('_')[0] for x in all_patient_names]))
                root = '/'.join(self.img_lst[2].values.tolist()[0].split('/')[:-1]) + '/'
              
                for patient_name in all_patient_names:

                    img_path = root + patient_name + '.npy'
                    patient_list = self.img_lst[self.img_lst[0].str.contains(patient_name)].copy()
                                                            
                    cand_ids = [ int(x.split('_')[-1]) for x in patient_list[0].values.tolist()]
                    labels = patient_list[1].values.tolist()

                    pairs = zip(cand_ids, labels)
                    img_raw = np.load(img_path).astype(np.uint8)

                    for idc, lab in pairs:
                        img = img_raw[idc:idc+1,:,:,:, :self.in_img_shape[4]].copy()

                        dict = {'data' : img,
                                'labels' : lab}

                        self.data_pairs.append(dict)

                        if self.importance_sampling:
                            self.index_list.append((index_counter, self.importance_dict[lab]))
                            index_counter += 1
                        else: 
                            self.index_list.append(index_counter)
                            index_counter += 1
                        

                        #if(len(self.data_pairs) % 1000 == 0):
                        print(len(self.data_pairs), "data records in RAM")
            if self.is_training:
                self.shuffle_index_lst_()

    def random_importance_sampled_indices(self, index_list, batchsize):
        indices = []
        cnt = 0
        while True:
            idx, prob = index_list[cnt]
            if prob > np.random.uniform(0.0,1.0):
                indices.append(idx)
                if len(indices) == batchsize:
                    break
            cnt += 1
            if(cnt >= len(index_list)):
                cnt = 0
        return indices

    def read_batch(self):
        if (not self.importance_sampling):
            batch_df = self.index_list[self.counter * self.batch_size : (self.counter + 1) * self.batch_size]
            if len(batch_df) < self.batch_size:
                batch_df_res = self.index_list[:(self.batch_size - len(batch_df))]
                batch_df = batch_df + batch_df_res
                self.shuffle_index_lst_()
                self.counter = -1
            elif (self.counter + 1) * self.batch_size == len(self.index_list):
                self.shuffle_index_lst_()
                self.counter = -1 
        else:
            self.shuffle_index_lst_()
            batch_df = self.random_importance_sampled_indices(self.index_list, self.batch_size)

        self.numpy_image_batch = []
        self.numpy_label_batch = []

        for idx in batch_df:

            tup = self.data_pairs[idx]

            # load img
            data = tup['data'].copy()
            data = self.AugmentData(data, None).copy()

            self.numpy_label_batch.append(tup['labels'])
            self.numpy_image_batch.append(data)


        self.numpy_image_batch = np.asarray(self.numpy_image_batch)
        self.numpy_label_batch = np.asarray(self.numpy_label_batch)

        self.numpy_label_batch = np.reshape(self.numpy_label_batch, [self.batch_size, -1])

        self.counter += 1


    def shuffle_index_lst_(self):
        np.random.shuffle(self.index_list)

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

    def AugmentData(self, data, specific_slice=None):
        self.tmp_candidates[:] = 0
        def rand_mirror(tmp):
            if 0 in self.rand_mirror_axes and np.random.randint(0, 2) == 1:
                tmp = np.flipud(tmp)
            if 1 in self.rand_mirror_axes and np.random.randint(0, 2) == 1:
                tmp = np.fliplr(tmp)
            if 2 in self.rand_mirror_axes and np.random.randint(0, 2) == 1:
                tmp = np.swapaxes(np.fliplr(np.swapaxes(tmp, 1, 2)), 1, 2)
            return tmp

        def get_rand_crop_starts(cand_shape, mode, shape):
            if mode == 'tr': # random crop
                z_crop_start = np.random.randint(self.rand_cropping_coords[0], self.rand_cropping_coords[1]-shape[1]+1)
                y_crop_start = np.random.randint(self.rand_cropping_coords[2], self.rand_cropping_coords[3]-shape[2]+1)
                x_crop_start = np.random.randint(self.rand_cropping_coords[4], self.rand_cropping_coords[5]-shape[3]+1)
            elif mode == 'va': # crop from centrum
                z_crop_start = (cand_shape[0]-shape[1])//2
                y_crop_start = (cand_shape[1]-shape[2])//2
                x_crop_start = (cand_shape[2]-shape[3])//2
            return [z_crop_start, y_crop_start, x_crop_start]

        def rand_crop(cand, rand_crop_starts, shape):
            return cand[rand_crop_starts[0]:rand_crop_starts[0]+shape[1],
                        rand_crop_starts[1]:rand_crop_starts[1]+shape[2],
                        rand_crop_starts[2]:rand_crop_starts[2]+shape[3],:]

        def cross_crop(cand):
            num_crossed_layers = self.num_crossed_layers
            if 0 in self.cross_axes:
                layer_start = cand.shape[0]//2-int(np.ceil(self.num_crossed_layers//2))
                self.tmp_crossed_layer[0:self.num_crossed_layers] = cand[layer_start:layer_start+self.num_crossed_layers,:,:,:]
            if 1 in self.cross_axes:
                layer_start = cand.shape[1]//2-int(np.ceil(self.num_crossed_layers//2))
                self.tmp_crossed_layer[self.num_crossed_layers if 0 in self.cross_axes else 0: 2*self.num_crossed_layers if 0 in self.cross_axes else self.num_crossed_layers] = np.swapaxes(cand[:, layer_start:layer_start+self.num_crossed_layers,:].copy(), 0, 1)
            if 2 in self.cross_axes:
                layer_start = cand.shape[2]//2-int(np.ceil(self.num_crossed_layers//2))
                self.tmp_crossed_layer[-self.num_crossed_layers:] = np.swapaxes(cand[:,:, layer_start:layer_start+self.num_crossed_layers].copy(), 0, 2)

        if self.is_training:
            # if rand cand use random num_candidates candidates
            #for cand_cnt, cand_id in enumerate(np.random.permutation(range(data.shape[0]))[0:self.num_candidates] if self.img_shape[0] != self.in_img_shape[0] else range(data.shape[0])[0:self.num_candidates]):
            for cand_cnt, cand_id in enumerate(np.random.permutation(range(data.shape[0]))[0:self.num_candidates] if self.img_shape[0] != self.in_img_shape[0] else range(data.shape[0])[0:self.num_candidates]):

                if self.cross_axes:
                    # get random crop starts
                    rand_crop_starts = get_rand_crop_starts(cand_shape=self.tmp_cand.shape, mode='tr', shape=[self.img_shape[2]]*4)
                    # get candidate
                    self.tmp_cand = rand_crop(data[cand_id,:,:,:,:], rand_crop_starts, [self.img_shape[2]]*4)
                else:
                    # get candidate
                    self.tmp_cand = data[cand_id,:,:,:,:]
                    # get random crop starts
                    rand_crop_starts = get_rand_crop_starts(cand_shape=self.tmp_cand.shape, mode='tr', shape=self.img_shape)

                # random rotation
                if self.rand_rot:
                    self.rand_angle_1 = np.random.randint(self.min_rot_angle, self.max_rot_angle)

                    # rotate rand_angle_1 around rectangular planes
                    if self.degree_90_rot:
                        rot90 = np.random.randint(0, 4)
                        self.rand_angle_1 += rot90*90

                    M = cv2.getRotationMatrix2D((data.shape[3]//2, data.shape[2]//2), self.rand_angle_1, 1)

                    # only rotate around one axis
                    self.rand_axis = np.random.permutation(self.rand_rot_axes)[0]
                    self.tmp_cand = rotate_3d(self.tmp_cand, M, self.rand_axis, self.rand_angle_1, rand_crop_starts, self.img_shape, self.cross_axes)

                if not self.cross_axes:
                    self.tmp_candidates[cand_cnt,:,:,:,:] = rand_crop(self.tmp_cand, rand_crop_starts, self.img_shape).astype(float)
                else:
                    # include cross_axes_cropping
                    cross_crop(self.tmp_cand)

                    if not self.rand_drop_planes:
                        self.tmp_candidates[cand_cnt,:,:,:,:] = self.tmp_crossed_layer.astype(float)
                    else:
                        rand_ids = np.random.randint(0, self.tmp_crossed_layer.shape[0], self.tmp_candidates.shape[1])
                        self.tmp_candidates[cand_cnt,:,:,:,:] = self.tmp_crossed_layer[rand_ids].astype(float)

                # random mirrors
                if self.rand_mirror_axes:
                    self.tmp_candidates[cand_cnt,:,:,:,:] = rand_mirror(self.tmp_candidates[cand_cnt,:,:,:,:])

        else:

            for cand_cnt, cand_id in enumerate(range(min(self.num_candidates, data.shape[0]))): #
                # get candidate
                self.tmp_cand = data[cand_id,:,:,:,:]
                # cropping from centrum
                rand_crop_starts = get_rand_crop_starts(cand_shape=self.tmp_cand.shape, mode='va', shape=self.img_shape if not self.cross_axes else [self.img_shape[2]]*4)               
                if not self.cross_axes:
                    if specific_slice == None:
                        self.tmp_candidates[cand_cnt,:,:,:,:] = rand_crop(self.tmp_cand, rand_crop_starts, self.img_shape).astype(float)
                    else:
                        self.tmp_candidates[cand_cnt,:,:,:,:] = self.tmp_cand[specific_slice:specific_slice+1, :, :, :].astype(float)

                else:
                    cross_crop(rand_crop(self.tmp_cand, rand_crop_starts, [self.img_shape[2]]*4))

                    if not self.rand_drop_planes:
                        self.tmp_candidates[cand_cnt,:,:,:,:] = self.tmp_crossed_layer.astype(float)
                    else:
                        rand_ids = np.random.randint(0, self.tmp_crossed_layer.shape[0], self.tmp_candidates.shape[1])
                        self.tmp_candidates[cand_cnt,:,:,:,:] = self.tmp_crossed_layer[rand_ids].astype(float)                  

        # normalize and zero_center_data
        self.tmp_candidates[:,:,:,:, 0] = ((self.tmp_candidates[:,:,:,:, 0]) * 1.0/255 - 0.25)
        # [0, 255] -> [0.0, 1.0] for prob_maps
        if self.tmp_candidates.shape[4] == 2:
            self.tmp_candidates[:,:,:,:, 1] = (self.tmp_candidates[:,:,:,:, 1] * 1.0/255)
        return self.tmp_candidates

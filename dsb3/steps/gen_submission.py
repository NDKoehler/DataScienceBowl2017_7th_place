import sys
import numpy as np
import pandas as pd
import os
from datetime import datetime
import tensorflow as tf
import json
import glob
import json
from collections import OrderedDict
from .. import pipeline as pipe
from .. import utils
from .. import tf_tools

import matplotlib.pyplot as plt
from tqdm import tqdm 

np.random.seed(17) # do NOT change
'''
step generats dsb3 predictions using a cancer_classifier. Num augmentations can be varied. First score calculated always has no augmentation.
'''

def run(splitting,
        checkpoint_dir,
        num_augs_per_img,
        patients_lst_path,
        sample_submission_lst_path
        ):

    # check dataset
    if pipe.dataset_name != 'dsb3':
        raise ValueError('Only possible for dataset dsb3')
        sys.exit()
    # check key in splitting
    if splitting not in ['submission', 'validation']:
        raise ValueError('splitting key {} not valid. Use one of these: [validation, submission]/'.format(splitting))
        sys.exit()
    # default value
    if not bool(patients_lst_path) :
        patients_lst_path = 'interpolate_candidates'
        pipe.log.info('use default patients_lst_path {}'.format(patients_lst_path))
    # get patiens data from within pipe/ called folder /arrays.*.npy
    if patients_lst_path == 'filter_candidates' or \
         patients_lst_path == 'interpolate_candidates':
                va_lst_path = pipe.get_step_dir(patients_lst_path) + 'va_patients.lst' if splitting=='validation' else 'ho_patients.lst'
                data_from_wihtin_pipe = 'fc' if patients_lst_path=='filter_candidates' else 'ic'
    else:
        # if not path column in called list is used
        data_from_wihtin_pipe = False
        data_DF = pd.read_csv(patients_lst_path, sep='\t', header=None)
        data_DF = data_DF.rename(columns={0:'id',1:'cancer_label',2:'path'})
    # laod list
    if splitting == 'submission':
        sample_submission = pd.read_csv(sample_submission_lst_path) # header: id, cancer
        patients_2_process = sample_submission['id'].values.tolist()
    elif splitting == 'validation':
        validation_DF = pd.read_csv(patients_lst_path, header=None, sep='\t')
        validation_DF = validation_DF.rename(columns={0:'id',1:'cancer_label',2:'path'})
        patients_2_process = validation_DF['id'].values.tolist()
    patients_2_process = list(set(patients_2_process))

    # initialize result container for predictions (=scores) and logloss (last one only is used for displaying information during calculation)
    if num_augs_per_img > 0:
        # first one is not_augmented
        patients_predictions = np.zeros((len(patients_2_process), num_augs_per_img+1), dtype=np.float32)
        arr = [False, True]
    else:
        patients_predictions = np.zeros((len(patients_2_process),1), dtype=np.float32)
        arr = [False]
    patients_losses = np.zeros((len(patients_2_process),2), dtype=np.float32)
    # loop over not_augmented, augmented
    for is_augmented in arr:
        if not is_augmented:
            batch_size_ = 1
            num_augs_per_img_ = 0
        else:
            batch_size_ = num_augs_per_img
            num_augs_per_img_ = num_augs_per_img
        # get some parameters for session
        net_config = {}
        net_config['gpu_fraction']   = pipe.GPU_memory_fraction
        max_batch_size = 64
        if batch_size_>max_batch_size:
            pipe.log.info('batch_size param is set from {} to max_batch_size {}'.format(batch_size_, max_batch_size))
        net_config['batch_size']     = min(batch_size_, max_batch_size) # set maximal batchsize
        net_config['checkpoint_dir'] = checkpoint_dir
        net_config['GPUs'] =  pipe.GPU_ids

        # generate cancer_score_session
        gen_cancer_score = cancer_score(num_augs_per_img_, net_config, splitting)

        labels = []
        # generate patients prediction
        for pa_cnt, patient in tqdm(enumerate(patients_2_process)):
            try:
                if data_from_wihtin_pipe:
                    candidates = pipe.load_array(patient + '.npy', patients_lst_path)
                else:
                    candidates = np.load(data_DF[data_DF['id']==patient]['path'].values.tolist()[0])
            except:
                pipe.log.error('could not load data_array of patient {}'.format(patient))
                continue
            if splitting == 'validation':
                lab = float(validation_DF[validation_DF['id']==patient]['cancer_label'].values.tolist()[0])
                labels.append(lab)
                scores, logl = gen_cancer_score.predict_score(candidates, lab)
            else:
                scores = gen_cancer_score.predict_score(candidates)
            if not is_augmented:
                patients_predictions[pa_cnt, 0] = scores[0]
                if splitting == 'validation':
                    patients_losses[pa_cnt, 0]  = logl
            else:
                patients_predictions[pa_cnt, 1:]  = scores
                if splitting == 'validation':
                    patients_losses[pa_cnt,1] = logl
        # print some pre info before long long augmentation run
        if splitting == 'validation':
            if not is_augmented:
                print ('logloss_no_augment', np.mean(patients_losses[:,0]))
            else:
                print ('logloss_augment', np.mean(patients_losses[:,1]))

    # mean over all predictions and save submission csv
    for pa_cnt, patient in tqdm(enumerate(patients_2_process)):
        if splitting == 'submission':
                sample_submission[sample_submission['id'] == patient]['cancer'] = np.mean(patients_predictions[pa_cnt])
    # save submission to submission dir
    if splitting == 'submission':
        submission_path = pipe.get_step_dir() + 'submission.csv'
        sample_submission.to_csv(submission_path, index=False)

    elif splitting == 'validation':
        labels = np.array(labels,dtype=np.float32)
        print('patients_predictions[:, 0].shape', patients_predictions[:, 0].shape)
        print('patients_losses[:, 0].shape', patients_losses[:, 0].shape)
        score_no_augment = np.mean(patients_losses[:,0])
        print ('score_no_augment', score_no_augment)
        scores_single = []
        scores_mean = []
        scores_mean_5050 = []

        for i in range(1, patients_predictions.shape[1]):
            score_single = logloss(patients_predictions[:,i], labels)
            scores_single.append(score_single)

            score_mean = logloss(np.mean(patients_predictions[:,:i], axis=1), labels)
            scores_mean.append(score_mean)

            score_mean_5050 = 0.5 * (logloss(np.mean(patients_predictions[:,1:i], axis=1), labels) + logloss(patients_predictions[:,0], labels))
            scores_mean_5050.append(score_mean_5050)

        if num_augs_per_img>0:
            print('score_single', score_single)
            print('score_mean', score_mean)
            print('score_mean_5050', score_mean_5050)

            plt.scatter(range(len(scores_single)), scores_single, label='single augmented scores')
            plt.plot(scores_mean_5050, label='score mean 5050')
            plt.plot(scores_mean, label='mean scores')
        plt.axhline(score_no_augment,0, len(scores_single)+1)
        plt.xlim(0,num_augs_per_img)

        plt.title(checkpoint_dir.split('/')[-1])
        plt.legend()
        plt.ylabel('validation score')
        plt.xlabel('number of augmentations')
        plt.savefig(pipe.get_step_dir() + '/score_over_num_augs.png')
        plt.show()
    # save some information about this submission
    with open(pipe.get_step_dir() + 'submission.info', 'w') as info:
        info.write('used checkpoint: {}\nnum augmentations: {}\npatients_lst_path: {}'.format(checkpoint_dir, num_augs_per_img, patients_lst_path))

def logloss(prediction, label):
    eps = 1e-7
    prediction = np.maximum(np.minimum(prediction, 1-eps), eps)
    return -np.mean(  label*np.log(prediction) + (1-label)*np.log(1-prediction) ) # eval formula from kaggle

class cancer_score():
    def __init__(self, num_augs_per_img, net_config, splitting):
        super(cancer_score, self).__init__()
        '''
        input:
            candidate with image_shape where in the first channel there is the scan and in the second the prob_map
            value_range [0, 255] as uint8
        output:
            nodule_score value range [0.0, 1.0]
        task:
            calculates the nodule_score by meaning augmented (optional) predictions
        ATTENTION:
            num_augs_per_img == 0 means that there is only one prediciton that is not augmented
        '''
        # define some class_variables
        self.splitting = splitting
        if net_config['GPUs'] is not None: os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in net_config['GPUs']])
        self.checkpoint_dir = net_config['checkpoint_dir']
        self.config = json.load(open(self.checkpoint_dir + '/config.json'))
        self.image_shape = self.config['image_shape']
        self.label_shape = self.config['label_shape']
        self.num_augs_per_img = num_augs_per_img
        self.batch_size  = net_config['batch_size']
        self.reuse = None
        # define some placeholders
        self.num_batches = int(np.ceil((max(1,self.num_augs_per_img))/self.batch_size))
        self.batch = np.zeros([self.batch_size] + self.image_shape, dtype=np.float32)
        self.all_predictions = np.zeros((max(1,self.num_augs_per_img)), dtype=np.float32)
        # laod session
        self.sess, self.pred_ops, self.data = tf_tools.load_network(
                image_shape=self.image_shape,
                checkpoint_dir=self.checkpoint_dir,
                reuse=self.reuse,
                batch_size=self.batch_size,
                )
        # get List_iterator for augmentation
        sys.path.append(self.checkpoint_dir)
        from io_modules.list_iterator_classification import List_Iterator
        self.pred_iter = List_Iterator(
                self.config,
                img_lst=False,
                img_shape=self.image_shape,
                label_shape=self.label_shape,
                batch_size=self.batch_size,
                shuffle=False,
                is_training=bool(self.num_augs_per_img)
                )

    def predict_score(self, candidates, lab=None):
        '''
        input:
            candidates with image_shape where in the first channel there is the scan and in the second the prob_map
            value_range [0, 255] as uint8
        output:
            cancer_score value range [0.0, 1.0]
        task:
            calculates the cancer_score by meaning augmented (optional) predictions
        '''
        # clean batch
        self.batch[:] = -0.25
        self.all_predictions[:] = 0.0
        for b_cnt in range(self.num_batches):
            for cnt in range(self.batch_size):
                if b_cnt*self.batch_size + cnt >= max(1,self.num_augs_per_img): break
                self.batch[cnt] = self.pred_iter.AugmentData(candidates.copy())
            predictions = self.sess.run(self.pred_ops, feed_dict = {self.data['images']: self.batch})['probs'][:,0]
            self.all_predictions[b_cnt*self.batch_size:min(max(self.num_augs_per_img,1),(b_cnt+1)*self.batch_size)] = \
                           predictions[:max(min(self.num_augs_per_img,1),
                                        (b_cnt+1)*self.batch_size)-b_cnt*self.batch_size]
        if self.splitting == 'validation':
            l = logloss(predictions, 1 if lab>0 else 0)
            return self.all_predictions, l
        else:
            return self.all_predictions

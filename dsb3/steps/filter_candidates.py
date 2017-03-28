'''
Filters numerous HR-candidates using the score from a neural nodule-no_nodule-classifier trained on LUNA16.
'''
import os, sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from collections import OrderedDict
from .. import pipeline as pipe
from .. import utils
from .. import tf_tools

def run(num_augs_per_img,
        checkpoint_dir,
        n_candidates,
        all_patients):
    # load some information json
    gen_candidates_json = pipe.load_json('out.json', 'gen_candidates')
    out_json = OrderedDict()
    # get some parameters
    net_config = {}
    net_config['gpu_fraction']   = pipe.GPU_memory_fraction
    batch_size = num_augs_per_img
    max_batch_size = 64
    if batch_size>max_batch_size:
        pipe.log.info('batch_size param is set from {} to max_batch_size {}'.format(batch_size, max_batch_size))
    net_config['batch_size']     = min(batch_size, max_batch_size) # set maximal batchsize
    net_config['checkpoint_dir'] = checkpoint_dir
    net_config['GPUs'] =  pipe.GPU_ids
    # generate nodule_score_session
    gen_nodule_score = score_nodules(num_augs_per_img, net_config, pipe.dataset_name )
    # loop over lsts
    if all_patients:
        splits = ['tr', 'va', 'ho']
    else:
        splits = ['va']
    for lst_type in splits:
        patients_DF_path = pipe.get_step_dir('interpolate_candidates') + lst_type + '_patients.lst'
        if os.path.exists(patients_DF_path):
            try:
                patients_DF = pd.read_csv(patients_DF_path, sep = '\t', header=None)
            except:
                pipe.log.error(' some error occured reading patients list {}. continue with next list.'.format(lst_type))
                continue
        else:
            pipe.log.error('patients list {} does not exist. continue with next list.'.format(lst_type))
            continue
        if len(patients_DF) == 0:
            pipe.log.error('empty patient list {}'.format(lst_type))
            continue
        # initialize out_list
        open(pipe.get_step_dir() + lst_type + '_candidates_filtered.lst', 'w').close()
        # get scores for all candidates of each patient
        for pa_cnt, patient in enumerate(tqdm(patients_DF[0].values.tolist())):
            # read and write jsons
            patient_json = gen_candidates_json[patient]
            out_json[patient] = patient_json.copy()
            out_patient_json = out_json[patient]
            clusters_json = patient_json['clusters']
            out_patient_json['clusters'] = []
            out_clusters_json = out_patient_json['clusters']
            num_real_candidates = len(clusters_json)
            # get patients interpolated candidates
            all_candidates = pipe.load_array(patient+'.npy', 'interpolate_candidates')[:num_real_candidates]
            # get candidates labels
            # get all candidates scores
            all_candidates_scores = []
            for cand_cnt, candidate in enumerate(all_candidates):
                # do not treat dummy candidates
                if cand_cnt >= num_real_candidates:
                    break
                candidate = np.expand_dims(candidate,0)
                if pipe.dataset_name == 'LUNA16':
                    lab     = float(clusters_json[cand_cnt]['nodule_priority'])
                    score, logloss = gen_nodule_score.predict_score(candidate, lab)
                else:
                    score = gen_nodule_score.predict_score(candidate)
                all_candidates_scores.append(score)
            # extract candidates with highest scores
            out_candidates_idx = list(np.argsort(np.array(all_candidates_scores)).copy()[::-1])
            out_candidates_idx = out_candidates_idx[:n_candidates]
            for idx_cnt, idx in enumerate(out_candidates_idx):
                if idx_cnt==0:
                    out_candidates = np.expand_dims(all_candidates[idx], axis=0)
                else:
                    out_candidates = np.vstack([out_candidates, np.expand_dims(all_candidates[idx], axis=0)])
            path = pipe.save_array(patient + '.npy', out_candidates)
            with open(pipe.get_step_dir() + lst_type + '_candidates_filtered.lst', 'a') as out_lst:
                for idx_cnt, idx in enumerate(out_candidates_idx):
                    if idx_cnt < num_real_candidates:
                        if pipe.dataset_name == 'LUNA16':
                            nodule_prio = clusters_json[idx]['nodule_priority']
                        nodule_score = all_candidates_scores[idx]
                        clusters_json[idx]['nodule_score'] = str(nodule_score)
                        out_clusters_json.append(clusters_json[idx])
                    else:
                        print ('DUMMY')
                        if pipe.dataset_name == 'LUNA16':
                            nodule_prio = 0
                        nodule_score = 0
                    if pipe.dataset_name == 'LUNA16':
                        out_lst.write('{}\t{}\t{}\t{}\n'.format(patient + '_' + str(idx_cnt), nodule_prio, path, str(nodule_score)))
                    else:
                        out_lst.write('{}\t{}\t{}\t{}\n'.format(patient + '_' + str(idx_cnt), patient_json['label'], path, str(nodule_score)))
        pipe.save_json('out.json', out_json)

class score_nodules():
    def __init__(self, num_augs_per_img, net_config, dataset_name):
        super(score_nodules, self).__init__()
        # define some class_variables
        self.dataset_name = dataset_name
        if net_config['GPUs'] is not None: os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in net_config['GPUs']])
        self.checkpoint_dir = net_config['checkpoint_dir']
        self.config = json.load(open(self.checkpoint_dir + '/config.json'))
        self.image_shape = self.config['image_shape']
        self.label_shape = self.config['label_shape']
        self.num_augs_per_img = num_augs_per_img
        self.batch_size  = net_config['batch_size']
        self.reuse = None
        # define some placeholders
        self.num_batches = int(np.ceil(self.num_augs_per_img/self.batch_size))
        self.batch = np.zeros([self.batch_size] + self.image_shape, dtype=np.float32)
        self.cand_predictions = np.zeros((self.num_augs_per_img), dtype=np.float32)
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
        self.pred_iter = List_Iterator(self.config, img_lst=False, img_shape=self.image_shape, label_shape=self.label_shape, batch_size=self.batch_size, shuffle=False, is_training=False if self.num_augs_per_img==1 else True)

    def logloss(self, prediction, label):
        eps = 1e-6
        prediction = np.maximum(np.minimum(prediction, 1-eps), eps)
        return -np.mean(  label*np.log(prediction) + (1-label)*np.log(1-prediction) ) # eval formula from kaggle

    def predict_score(self, candidate, lab=None):
        '''
        input:
            candidate with image_shape where in the first channel there is the scan and in the second the prob_map
            value_range [0, 255] as uint8
        output:
            nodule_score value range [0.0, 1.0]
        task:
            calculates the nodule_score by meaning augmented (optional) predictions
        ATTENTION:
            num_augs_per_img == 1 means that first prediction is not augmented!
        '''
        # clean batch
        self.batch[:] = -0.25
        self.cand_predictions[:] = 0.0
        for b_cnt in range(self.num_batches):
            for cnt in range(self.batch_size):
                if b_cnt*self.batch_size + cnt >= self.num_augs_per_img:
                    break
                self.batch[cnt] = self.pred_iter.AugmentData(candidate.copy())
            predictions = self.sess.run(self.pred_ops, feed_dict = {self.data['images']: self.batch})['probs']
            self.cand_predictions[b_cnt*self.batch_size:min(self.num_augs_per_img,(b_cnt+1)*self.batch_size)] = \
                           predictions[:min(self.num_augs_per_img,
                                        (b_cnt+1)*self.batch_size)-b_cnt*self.batch_size, 0]
        # do NOT use mean here due to the case that the final batch can have dummies
        total_prediction = np.sum(self.cand_predictions)/self.num_augs_per_img
        if self.dataset_name == 'LUNA16':
            logloss = self.logloss(total_prediction, 1 if lab>0 else 0)
            return total_prediction, logloss
        else:
            return total_prediction


import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

def run(batch_size, checkpoint_dir):
    net_config['gpu_fraction']  = 0.85
    net_config['batch_size']     = num_augs_per_img
    net_config['checkpoint_dir'] =  './checkpoints/luna_candidate_level_new_equaldist'

    # generate nodule_score_session
    gen_nodule_score = score_nodules(num_augs_per_img, net_config, dataset_name)

    for lst in ['va_candidates.lst']:#['tr_candidates.lst', 'va_candidates.lst']:
        # cols: patient, label, path to npy
        candidates_lst =  pd.read_csv(H['data_in_dir'] + lst, sep='\t', header=None)
        candidates_lst['prediction'] = candidates_lst[1].copy()
        print (lst, 'list length: ', len(candidates_lst))

        loglosses = []
        predictions = {0:[], 1:[], 2:[], 3:[], 4:[]}

        for cnt, row in enumerate(tqdm(candidates_lst.iterrows())):
            patient = row[1][0]
            lab     = float(row[1][1])
            path    = row[1][2]

            candidate = np.load(path)
            # print ('score: {}, label: {}'.format(score, lab))
            if dataset_name == 'LUNA16':
                score, logloss = gen_nodule_score.predict_score(candidate)
                loglosses.append(logloss)
                predictions[lab].append(score)
            else:
                score = gen_nodule_score.predict_score(candidate)
                predictions['all'].append(score)
            candidates_lst['prediction'].loc[candidates_lst[0]==patient] = np.mean(score)

        candidates_lst.to_csv(H['data_in_dir'] + lst.replace('.lst','_pred.lst'), header=None, sep='\t')

        if dataset_name == 'LUNA16':
            all_data = []
            print ('LOGLOSS: ', np.mean(loglosses))

            for i in range(5):
                if len(predictions[i])>1:
                    sns.distplot(np.array(predictions[i]), 50, label='prio '+str(i))
        else:
            print (lst, 'MEAN PREDICTIONS: ', np.mean(predictions['all']))
            if len(predictions['dsb3'])>1:
                sns.distplot(np.array(predictions['all']), 50, label='dsb3 {}'.format(lst))

        plt.xlim([0,1])
        plt.legend()
        plt.savefig(H['data_in_dir'] + lst + '_predictions_hist.png')
        plt.close()

class score_nodules():
    def __init__(self, num_augs_per_img, net_config, dataset_name):
        super(score_nodules, self).__init__()

        self.dataset_name = dataset_name

        if net_config['GPUs'] is not None: os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in net_config['GPUs']])
        self.checkpoint_dir = net_config['checkpoint_dir']
        sys.path.append(self.checkpoint_dir + '/')
        from io_modules.list_iterator_classification import List_Iterator
        sys.path.append(self.checkpoint_dir + '/architectures/')
        from tools.basic_logging import initialize_logger
        import architectures.resnet2D as resnet2D

        self.config = json.load(open(self.checkpoint_dir + '/config.json'))
        self.image_shape = self.config['image_shape']
        self.label_shape = self.config['label_shape']
        self.num_augs_per_img = num_augs_per_img

        self.batch_size  = net_config['batch_size']
        self.reuse = None
        self.num_batches = int(np.ceil(self.num_augs_per_img/self.batch_size))
        self.batch = np.zeros([self.batch_size] + self.image_shape, dtype=np.float32)
        self.cand_predictions = np.zeros((self.num_augs_per_img), dtype=np.float32)

        self.sess, self.pred_ops, self.data = load_network(
                config=self.config,
                net=resnet2D.resnet2D,
                initialize_logger=initialize_logger,
                net_name=self.config['model_name'],
                image_shape=self.image_shape,
                checkpoint_dir=self.checkpoint_dir,
                gpu_fraction=net_config['gpu_fraction'],
                GPU = net_config['GPUs'][0],
                batch_size=self.batch_size,
                reuse=self.reuse)

        self.pred_iter = List_Iterator(self.config, img_lst=False, img_shape=self.image_shape, label_shape=self.label_shape, batch_size=self.batch_size, shuffle=False, is_training=False if self.num_augs_per_img==1 else True)

    def logloss(self, prediction, label):
        eps = 1e-6
        prediction = np.maximum(np.minimum(prediction, 1-eps), eps)
        return -np.mean(  label*np.log(prediction) + (1-label)*np.log(1-prediction) ) # eval formula from kaggle

    def predict_score(self, candidate):
        '''
        input:
            candidate with image_shape where in the first channel there is the scan and in the second the prob_map
            value_range [0, 255] as uint8
        output:
            nodule_score value range [0.0, 1.0]
        task:
            calculates the nodule_score by meaning augmented (optional) predictions
        '''
        # clean batch
        self.batch[:] = -0.25
        self.cand_predictions[:] = 0.0

        for b_cnt in range(self.num_batches):
            for cnt in range(self.batch_size):
                if b_cnt*self.batch_size + cnt >= self.num_augs_per_img: break

                self.batch[cnt] = self.pred_iter.AugmentData(candidate.copy())

            predictions = self.sess.run(self.pred_ops, feed_dict = {self.data['images']: self.batch})['predictions']
            self.cand_predictions[b_cnt*self.batch_size:min(self.num_augs_per_img,(b_cnt+1)*self.batch_size)] = \
                           predictions[:min(self.num_augs_per_img,
                                        (b_cnt+1)*self.batch_size)-b_cnt*self.batch_size, 0]
        total_prediction = np.mean(self.cand_predictions)
        if self.dataset_name == 'LUNA16':
            logloss = self.logloss(total_prediction, 1 if lab>0 else 0)
            return total_prediction, logloss
        else:
            return total_prediction


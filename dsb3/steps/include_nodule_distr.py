import numpy as np
import json
from collections import OrderedDict
from scipy.spatial import distance
from numpy.linalg import eigh
from matplotlib import pyplot as plt
import sklearn.metrics
import xgboost as xgb
from .. import pipeline as pipe
import numpy.random as random
import sys
import pandas as pd
np.random.seed(132)

def run_xgboost(dtrain, dtest):
    xg_params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric" : "logloss",
        "eta": random.uniform(0.01, 0.3),
        "max_depth": random.randint(2, 4),
        "subsample": random.uniform(0.5, 0.95),
        "colsample_bytree": random.uniform(0.5, 0.95),
        "silent": 1,
        "seed": 0,
        "nthread" : 5
   }
    num_boost_round = 1000
    early_stopping_rounds = 25  
    evallist = [(dtest, 'test')]
    
    bst = xgb.train(xg_params, dtrain, 1000, evals=evallist, early_stopping_rounds = 20)
    
    #calculating predcition not necessary, score already saved
    #predictions = bst.predict(dtest, ntree_limit = bst.best_ntree_limit)

    #uses early stopping to determine optimal epoch
    log_loss = bst.best_score

    return log_loss, xg_params, bst

def logloss(prediction, label):
    eps = 1e-7
    prediction = np.maximum(np.minimum(prediction, 1-eps), eps)
    return -np.mean(  label*np.log(prediction) + (1-label)*np.log(1-prediction) ) # eval formula from kaggle

def sort_and_reverse_1Darray(array):
    return np.sort(array)[::-1]

def load_data(lst_path, n_candidates):
    # load lst
    candidates_DF = pd.read_csv(lst_path, sep='\t', header=None)
    candidates_DF = candidates_DF.rename(columns={0:'id',1:'label',2:'xxx', 3:'cand_score'})

    patients_lst = list(np.random.permutation(list(set(candidates_DF['id'].str.split('_').str[0].values.tolist()))))
    all_candidates_scores = np.zeros((len(patients_lst), n_candidates), dtype=np.float32)
    all_labels = np.zeros((len(patients_lst)), dtype=np.float32)
    for pa_cnt, patient in enumerate(patients_lst):
        patient_scores = sort_and_reverse_1Darray(candidates_DF[candidates_DF['id'].str.split('_').str[0]==patient]['cand_score'].values)[:n_candidates]
        labels = candidates_DF[candidates_DF['id'].str.split('_').str[0]==patient]['label'].values.tolist()
        if len(list(set(labels)))!= 1:
            print ('ERROR!!! with labels')
        all_labels[pa_cnt] = labels[0]
        all_candidates_scores[pa_cnt,:len(patient_scores)] = patient_scores.copy()
    return np.array(patients_lst), all_candidates_scores, all_labels

def run(lists_to_predict, n_candidates=20, bin_size=0.05, kernel_width=0.2, xg_max_depth=2, xg_eta=1, xg_num_round=2, sample_submission_lst_path=None):
    splits = [lst2pred.split('/')[-1].split('.')[0].split('_')[0] for lst2pred in lists_to_predict]
    data = {}
    for lst_cnt, lst_type in enumerate(splits):
        data[lst_type] = {}
        patients_lst, all_candidates_scores, all_labels = load_data(lists_to_predict[lst_cnt],n_candidates)
        data[lst_type]['patients'] = patients_lst.copy()
        data[lst_type]['all_candidates_scores'] = all_candidates_scores.copy()
        data[lst_type]['all_labels'] = all_labels.copy()
    dtrain = xgb.DMatrix(data['tr']['all_candidates_scores'].copy(), label=data['tr']['all_labels'].copy())
    dtest  = xgb.DMatrix(data['va']['all_candidates_scores'].copy(), label=data['va']['all_labels'].copy())   

    scores_params_bsts = []
    for i in range(1000):
        score_params_bst = run_xgboost(dtrain, dtest)
        scores_params_bsts.append(score_params_bst)
    
    sorted_by_score = sorted(scores_params_bsts, key=lambda tup: tup[0])
    print("best_score: ", sorted_by_score[0][0])
    print("best params: ", sorted_by_score[0][1])
    best_bst = sorted_by_score[0][2]

    print ('-----------------------')
    # predict validation_lst
    predictions_va = best_bst.predict(dtest, ntree_limit = best_bst.best_ntree_limit)
    logloss_va = logloss(np.array(predictions_va),data['va']['all_labels'])
    print ('\nvalidation_logloss',logloss_va)
    #print ('pred_va',predictions_va)

    print ('-----------------------')
    # predict holdout
    sample_submission = pd.read_csv(sample_submission_lst_path)
    dpred  = xgb.DMatrix(data['ho']['all_candidates_scores'].copy())
    predictions_sub = best_bst.predict(dpred, ntree_limit = best_bst.best_ntree_limit)
    for patient in sample_submission['id'].values.tolist():
        sample_submission['cancer'][sample_submission['id']==patient] = float(predictions_sub[data['ho']['patients']==patient][0])
    sample_submission.to_csv('submission.csv',index=False,columns=['id','cancer'])
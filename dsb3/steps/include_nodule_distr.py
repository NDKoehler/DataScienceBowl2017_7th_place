import numpy as np
import json
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import distance
from numpy.linalg import eigh
import sklearn.metrics
import xgboost as xgb
from .. import pipeline as pipe

def run(n_candidates=20, bin_size=0.05, kernel_width=0.2, xg_max_depth=2, xg_eta=1, xg_num_round=2):
    resample_lungs_out = pipe.load_json('out.json', 'resample_lungs')
    gen_candidates_out = pipe.load_json('out.json', 'gen_candidates')

    n_pairs = int(n_candidates * (n_candidates-1)/2)
    patients = []
    labels = []
    nodule_weights = []
    nodule_dists_mutual = []
    nodule_dists_evalues = []
    nodule_dists_from_lung = []
    for patient, patient_json in gen_candidates_out.items():
        patients.append(patient)
        # cancer label
        labels.append(patient_json['label'])
        # nodule info
        positions = []
        weights = []
        for cluster in patient_json['clusters'][:n_candidates]:
            positions.append(cluster['center_px'])
            weights.append(cluster['prob_sum_cluster'])
        # nodule weights
        weights = np.array(sorted(weights, reverse=True)) # high-weight nodules to the front
        if weights.shape[0] < n_candidates: # fill up the last places with zeros NOTE that filling up the first places lead to a better result, but is inconsistent
            weights = np.concatenate((weights, np.zeros(n_candidates - weights.shape[0])))
        nodule_weights.append(weights)
        # nodule positions
        positions = np.array(positions)
        lung_shape = np.array(resample_lungs_out[patient]['resampled_scan_shape_zyx_px'])
        positions = 2*positions/lung_shape[None, :] - 1 # coordinate system in the center of the lung, distance to each boundary is 1
        # distance from lung boundaries
        dist_from_lung = 1 - np.max(np.abs(positions), axis=1)
        dist_from_lung = np.histogram(dist_from_lung, bins=np.arange(0, 1 + bin_size, bin_size))[0] / len(positions) # maximal value is 1, normalize with number of clusters
        nodule_dists_from_lung.append(dist_from_lung)
        # mutual distance
        dist_mutual = distance.pdist(positions)
        dist_mutual = np.histogram(dist_mutual, bins=np.arange(0, 2*np.sqrt(2) + bin_size, bin_size))[0] / len(positions) # maximal value is diagonal 2 * sqrt(2)
        nodule_dists_mutual.append(dist_mutual)
        # kernel and eigenvalues of distance matrix
        evalues, _ = eigh(np.exp(-distance.squareform(distance.pdist(positions))/kernel_width)) # simply an exponetial decrease, eigenvalues in ascending order
        if evalues.shape[0] < n_candidates:
            evalues = np.concatenate((np.zeros(n_candidates - evalues.shape[0]), evalues))
        nodule_dists_evalues.append(evalues)

    patients = np.array(patients)
    labels = np.array(labels)
    nodule_weights = np.array(nodule_weights)
    nodule_dists_mutual = np.array(nodule_dists_mutual)
    nodule_dists_from_lung = np.array(nodule_dists_from_lung)
    nodule_dists_evalues = np.array(nodule_dists_evalues)
    # try out different combinations of features
    # X = np.concatenate((nodule_weights, nodule_dists_from_lung, nodule_dists_mutual, nodule_dists_evalues), axis=1)
    # X = nodule_dists_evalues
    X = nodule_weights
    # X = nodule_dists_from_lung

    # choose validation and training sets
    tr = np.in1d(patients, np.array(pipe.patients_by_split['tr']))
    va = np.in1d(patients, np.array(pipe.patients_by_split['va'])) # validation set
    # va = np.in1d(patients, np.array(pipe.patients_by_label[-1])) # submission set

    X_train = X[tr]
    X_test = X[va]
    y_train = labels[tr]
    y_test = labels[va]

    # using xgboost
    xg_params = {'max_depth':xg_max_depth, 'eta':xg_eta, 'silent':1, 'objective':'binary:logistic'}
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    bst = xgb.train(xg_params, dtrain, xg_num_round)
    predictions = bst.predict(dtest)
    if y_test[0] != -1: # if we have a test set with known labels
        log_loss = sklearn.metrics.log_loss(y_test, predictions)
        print('xgboost')
        print(log_loss)
    else:
        log_loss = None
        import pandas as pd
        sample_submission = pd.read_csv('/'.join(pipe.raw_data_dir.split('/')[:-2]) + '/stage1_sample_submission.csv') # header: id, cancer
        for pa_cnt, patient in enumerate(pipe.patients_by_label[-1]):
            sample_submission.loc[pa_cnt, 'cancer'] = predictions[pa_cnt]
        sample_submission.to_csv(pipe.get_step_dir() + 'submission.csv', index=False)

    # using sklearn
    if False:
        lr = LogisticRegression()
        rfc = RandomForestClassifier(n_estimators=1000)
        for clf, name in [(lr, 'sklearn logistic regression'),
                          (rfc, 'sklearn random forest')]:
            clf.fit(X_train, y_train)
            if hasattr(clf, 'predict_proba'):
                predictions = clf.predict_proba(X_test)[:, 1]
            else:  # use decision function
                predictions = clf.decision_function(X_test)
                predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
            print(name)
            print(log_loss(y_test, predictions))

    return log_loss

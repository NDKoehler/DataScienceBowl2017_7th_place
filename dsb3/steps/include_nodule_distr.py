import numpy as np
from collections import OrderedDict
import json
import matplotlib.pyplot as pl
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import distance
from numpy.linalg import eigh
from sklearn.metrics import log_loss
import xgboost as xgb
from .. import pipeline as pipe

def run(**params):
    resample_lungs_out = pipe.load_json('out.json', 'resample_lungs')
    gen_candidates_out = pipe.load_json('out.json', 'gen_candidates')

    n_candidates = 20
    n_dim = int(n_candidates * (n_candidates - 1) / 2)
    patients = []
    labels = []
    nodule_dists = []
    nodule_dists_evalues = []
    nodule_weights = []
    nodule_dists_from_lung = []
    for patient, patient_json in gen_candidates_out.items():
        patients.append(patient)
        # cancer label
        labels.append(patient_json['label'])
        # nodule info
        nodule_pos = []
        nodule_weight = []
        for cluster in patient_json['clusters']:
            nodule_pos.append(cluster['center_px'])
            nodule_weight.append(cluster['prob_sum_cluster'])
        # nodule weights
        nodule_weight = np.array(sorted(nodule_weight, reverse=True))
        if nodule_weight.shape[0] < n_candidates:
            nodule_weight = np.concatenate((np.zeros(n_candidates - nodule_weight.shape[0]), nodule_weight))
        nodule_weights.append(nodule_weight)
        # nodule positions
        nodule_pos = np.array(nodule_pos)
        lung_shape = np.array(resample_lungs_out[patient]['resampled_scan_shape_zyx_px'])
        nodule_pos = 2 * nodule_pos / lung_shape[None, :] - 1 # coordinate system in the center of the lung
        # distance from lung boundaries
        dist_from_lung = 1 - np.max(np.abs(nodule_pos), axis=1)
        dist_from_lung = np.histogram(dist_from_lung, bins=np.arange(0, 1.2, 0.05))[0] / len(dist_from_lung)
        nodule_dists_from_lung.append(dist_from_lung)
        # distance from each other
        nodule_dist = distance.pdist(nodule_pos)
        nodule_dist = np.histogram(nodule_dist, bins=np.arange(0, 2, 0.05))[0] / len(nodule_dist)
        nodule_dists.append(nodule_dist)
        # kernel and eigenvalues of distance matrix
        #     nodule_dist_kernel = np.exp(-distance.squareform(nodule_dist)/0.2)
        #     nodule_dist_evalues, _ = eigh(nodule_dist_kernel)
        #     nodule_dist.sort()
        #     if nodule_dist.shape[0] < n_dim:
        #         nodule_dist = np.concatenate((np.zeros(n_dim - nodule_dist.shape[0]), nodule_dist))
        #     if nodule_dist_evalues.shape[0] < n_candidates:
        #         nodule_dist_evalues = np.concatenate((np.zeros(n_candidates - nodule_dist_evalues.shape[0]), nodule_dist_evalues))
        #     nodule_dists_evalues.append(nodule_dist_evalues)

    patients = np.array(patients)
    labels = np.array(labels)
    nodule_dists = np.array(nodule_dists)
    nodule_weights = np.array(nodule_weights)
    nodule_dists_from_lung = np.array(nodule_dists_from_lung)
    # nodule_dists_evalues = np.array(nodule_dists_evalues)
    X = np.concatenate((nodule_weights, nodule_dists_from_lung, nodule_dists), axis=1)
    # X = nodule_dists_evalues
    # X = nodule_weights
    # X = nodule_dists_from_lung

    tr = np.in1d(patients, np.array(pipe.patients_by_split['tr']))
    va = np.in1d(patients, np.array(pipe.patients_by_split['va']))
    
    X_train = X[tr]
    X_test = X[va]
    y_train = labels[tr]
    y_test = labels[va]

    # using xgboost
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    num_round = 2
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    bst = xgb.train(param, dtrain, num_round)
    prob_pos = bst.predict(dtest)
    print('xgboost')
    print(log_loss(y_test, prob_pos))

    # using sklearn
    # lr = LogisticRegression()
    # rfc = RandomForestClassifier(n_estimators=1000)
    # for clf, name in [(lr, 'sklearn logistic regression'),
    #                   (rfc, 'sklearn random forest')]:
    #     clf.fit(X_train, y_train)
    #     if hasattr(clf, 'predict_proba'):
    #         prob_pos = clf.predict_proba(X_test)[:, 1]
    #     else:  # use decision function
    #         prob_pos = clf.decision_function(X_test)
    #         prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    #     print(name)
    #     print(log_loss(y_test, prob_pos))

    # Y = TSNE().fit_transform(X_train)
    # pl.scatter(Y[:, 0], Y[:, 1], c=y_train)
    # pl.savefig('tsne.png')
    # pl.show()

    # Y = PCA().fit_transform(X_train)
    # pl.scatter(Y[:, 0], Y[:, 1], c=y_train)
    # pl.savefig('pca.png')
    # pl.show()

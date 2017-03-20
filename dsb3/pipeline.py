import os
from collections import OrderedDict
import logging
import numpy as np
from . import utils
from . import hrjson as json

avail_dataset_names = ['LUNA16', 'dsb3']

dataset_name = None
"""Either LUNA16 or dsb3."""

write_dir = None
"""Output directory were all processed data is written."""

step_dir = None
"""Output directory were all processed data of a specifc step is written.
Is subdirectory of `write_dir`."""

step_dir_data = None
"""Data directory of step.
Is subdirectory of `step_dir`."""

patients = None
"""List of all patients."""

patients_dict = None
"""Ordered dictionary that stores for each patient id an empty dict."""

patients_raw_data_path = None
"""Ordered dictionary that stores for each patient id the path to the raw data."""

logger = None
"""Global logger for the whole pipeline."""

# technical parameters
n_CPUs = 1
GPU_memory_fraction = 0.85

def init_pipe(config, n_patients_to_process=0):
    # simply passing the config is not very elegant, but
    # explicit names collide with the global variables
    if config['dataset_name'] not in avail_dataset_names:
        raise ValueError('dataset_name needs to be one of' + str(avail_dataset_names))
    global dataset_name
    dataset_name = config['dataset_name']
    if dataset_name == 'LUNA16':
        dataset_dir = config['dataset_dir_LUNA16']
    else:
        dataset_dir = config['dataset_dir_dsb3']
    global write_dir
    write_dir = config['write_basedir'].rstrip('/') + '/' + dataset_name + '/'
    init_patients(dataset_dir, n_patients_to_process)
    # logger for the whole pipeline
    global logger
    logger = utils.get_logger(write_dir + 'pipe.log', mode='a')
    # technical parameters
    global n_CPUs
    n_CPUs = config['n_CPUs']
    global GPU_memory_fraction
    GPU_memory_fraction = config['GPU_memory_fraction']
    # random seed
    np.random.seed(config['random_seed'])

def init_step(step_name):
    # logger
    logger.info('running step ' + step_name)
    # create directories, log files etc.
    global step_dir, step_dir_data
    step_dir, step_dir_data = get_step_dir(step_name)
    # data directory of step
    utils.ensure_dir(step_dir_data)

def save_step(dic, step_name=None):
    step_dir_ = step_dir if step_name is None else write_dir + step_name + '/'
    json.dump(dic, open(step_dir_ + 'out.json', 'w'), indent=4, indent_to_level=1)
    print('wrote', step_dir_ + 'out.json')

def load_step(step_name=None):
    step_dir_ = step_dir if step_name is None else write_dir + step_name + '/'
    return json.load(open(step_dir_ + 'out.json'))

def get_step_dir(step_name):
    step_dir = write_dir + step_name + '/'
    step_dir_data = write_dir + step_name + '/data/'
    return step_dir, step_dir_data

def init_patients(dataset_dir, n_patients_to_process=0):
    filename = write_dir + 'patients_raw_data_paths.json'
    global patients_raw_data_paths
    global patients
    if os.path.exists(filename):
        print('reading', filename)
        patients_raw_data_paths = json.load(open(filename), object_pairs_hook=OrderedDict)
        patients = list(patients_raw_data_paths.keys())
    else:
        from glob import glob
        from natsort import natsorted
        if dataset_name == 'LUNA16':
            patient_paths = glob(dataset_dir + 'subset*/*.mhd')
        elif dataset_name == 'dsb3':
            patient_paths = glob(dataset_dir + '*/')
        patient_paths = natsorted(patient_paths, key=lambda p: p.split('/')[-1])
        if dataset_name == 'LUNA16':
            patients = [p.split('/')[-1].split('.')[-2] for p in patient_paths]
        elif dataset_name == 'dsb3':
            patients = [p.split('/')[-2] for p in patient_paths]
        patients_raw_data_paths = OrderedDict(zip(patients, patient_paths))
        utils.ensure_dir(filename)
        json.dump(patients_raw_data_paths, open(filename, 'w'), indent=4)
        print('wrote', filename)
    global patients_dict
    patients_dict = OrderedDict([(patient, {}) for patient in patients_raw_data_paths])
    if n_patients_to_process > 0:
        patients = patients[:n_patients_to_process]
        patients_dict = OrderedDict(list(patients_dict.items())[:n_patients_to_process])
        patients_raw_data_paths = OrderedDict(list(patients_raw_data_paths.items())[:n_patients_to_process])
    print('... processing', len(patients), 'patients')

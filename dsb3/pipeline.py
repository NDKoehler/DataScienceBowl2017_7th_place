import os
from collections import OrderedDict
import json
import logging
from . import utils

dataset_name = None
"""Either LUNA16 or dsb3.
"""
write_dir = None
"""Output directory were all processed data is written.
"""
step_dir = None
"""Output directory were all processed data of a specifc step is written.

Is subdirectory of `write_dir`.
"""
step_dir_data = None
"""Data directory of step.

Is subdirectory of `step_dir`.
"""
patients = None
"""Ordered dictionary that stores patients and paths to raw data.
"""
logger = None
"""Global logger for the whole pipeline.
"""

# technical parameters
n_CPUs = 7
GPU_memory_fraction = 0.85

def init(config):
    if config['dataset_name'] not in ['LUNA16', 'dsb3']:
        raise ValueError('dataset_name needs to be one of' + str(['LUNA16', 'dsb3']))
    global dataset_name
    dataset_name = config['dataset_name']
    global write_dir
    write_dir = config['write_basedir'].rstrip('/') + '/' + config['dataset_name'] + '/'
    init_patients(config)
    # logger for the whole pipeline
    global logger
    logger = utils.get_logger(write_dir + 'pipe.log', mode='a')
    # technical parameters
    global n_CPUs
    n_CPUs = config['n_CPUs']
    global GPU_memory_fraction
    GPU_memory_fraction = config['GPU_memory_fraction']

def init_step(step_key):
    # logger
    logger.info('running step ' + step_key)
    # create directories, log files etc.
    global step_dir, step_dir_data
    step_dir = write_dir + step_key + '/'
    step_dir_data = write_dir + step_key + '/data/'
    # data directory of step
    utils.ensure_dir(step_dir_data)

def init_patients(config):
    filename = write_dir + 'patients_raw_scan_paths.json'
    global patients
    if os.path.exists(filename):
        print('reading', filename)
        patients = json.load(open(filename), object_pairs_hook=OrderedDict)
    else:
        from glob import glob
        from natsort import natsorted
        if dataset_name == 'LUNA16':
            print('processing LUNA16')
            patient_paths = glob(config['dataset_dir_LUNA16'] + 'subset*/*.mhd')
            file_type = '.mhd'
        elif dataset_name == 'dsb3':
            print('processing dsb3')
            patient_paths = glob(config['dataset_dir_dsb3'] + '*/')
            file_type = '.dcom'
        patient_paths = natsorted(patient_paths, key=lambda p: p.split('/')[-1])
        if dataset_name == 'LUNA16':        
            patient_ids = [p.split('/')[-1].split('.')[-2] for p in patient_paths]
        elif dataset_name == 'dsb3':
            patient_ids = [p.split('/')[-2] for p in patient_paths]            
        patients = OrderedDict(zip(patient_ids, patient_paths))
        utils.ensure_dir(filename)
        json.dump(patients, open(filename, 'w'), indent=4)
        print('wrote', filename)
    if config['n_patients_to_process'] > 0:
        patients = OrderedDict(list(patients.items())[:config['n_patients_to_process']])

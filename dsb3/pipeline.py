"""
Pipeline variables and functions.
"""
import os
import logging
import time
import numpy as np
from importlib import import_module
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

n_patients = None
"""Number of patients."""

patients = None
"""List of all patients. Use this to iterate over patients."""

patients_raw_data_path = None
"""Ordered dictionary that stores for each patient id the path to the raw data."""

# logging
log_pipe = None
"""Global logger for the whole pipeline.
Global info on pipeline usage, step runtimes and errors in steps automatically
go here. File opened in append mode."""

log_step = None
log = log_step # just a convenience name for authors of step modules
"""Step logger for the whole pipeline.
Everything about a specific step goes here. File opened in write mode."""

# technical parameters
n_CPUs = 1
"""Number of CPUs to use in multi-threading."""

GPU_ids = None
"""List of GPU ids for computations."""

GPU_memory_fraction = 0.85
"""Fraction of memory attributed to GPU computation."""

def run_step(step_name, params):
    log_pipe.info('run step ' + step_name)
    # output params dict for visual check
    print('run step', step_name, 'using params')
    for key, value in params.items():
        print('    {} = {}'.format(key, value))
    # create step directories, log files etc.
    global step_dir
    step_dir = get_step_dir(step_name)
    # data directory of step
    utils.ensure_dir(step_dir + '/data/')
    utils.ensure_dir(step_dir + '/figs/')
    # saving params dict
    json.dump(params, open(step_dir + 'params.json', 'w'), indent=4, indent_to_level=0)
    print('wrote', step_dir + 'params.json')
    # init step logger
    init_log_step(step_name)
    log.error('test')
    # register start time
    start_time = time.process_time()
    # import step module
    step = import_module('.steps.' + step_name, 'dsb3')
    try:
        step_dict = step.run(**params)
    except TypeError as e:
        if 'run() got an unexpected keyword argument' in str(e):
            raise TypeError(str(e) + '\n Provide one of the valid parameters\n' + step.run.__doc__)
        else:
            raise e
    if step_dict is not None:
        save_step(step_dict, step_name)
    log_pipe.info('finished after ' + time.strftime('%H:%M:%S', time.gmtime(time.process_time() - start_time)))

def save_step(d, step_name=None):
    step_dir_ = step_dir if step_name is None else write_dir + step_name + '/'
    json.dump(d, open(step_dir_ + 'out.json', 'w'), indent=4, indent_to_level=1)
    print('wrote', step_dir_ + 'out.json')

def load_step(step_name=None):
    step_dir_ = step_dir if step_name is None else write_dir + step_name + '/'
    return json.load(open(step_dir_ + 'out.json'))

def get_step_dir(step_name):
    step_dir = write_dir + step_name + '/'
    return step_dir

def init_patients():
    from collections import OrderedDict
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
    global n_patients
    if n_patients > 0:
        patients = patients[:n_patients]
        patients_raw_data_paths = OrderedDict(list(patients_raw_data_paths.items())[:n_patients])
    else:
        n_patients = len(patients)
    print('... processing', n_patients, 'patients')

def init_log(level=logging.INFO):
    global log_pipe
    filename = write_dir + 'log.txt'
    log_pipe = logging.getLogger(filename)
    log_pipe.setLevel(level) # it's necessary to set the level also here
    log_pipe= _add_file_handle_to_log(log_pipe, filename, 'a', level)
    open(filename, 'a').write('\n')

def init_log_step(step_name, level=logging.INFO):
    global log_step, log
    filename = step_dir + 'log.txt'
    log_step = logging.getLogger(filename)
    log_step.setLevel(level) # it's necessary to set the level also here
    log_step = _add_file_handle_to_log(log_step, filename, 'w', level)
    # write errors also to pipeline log file
    filename = write_dir + 'log.txt'
    log_step = _add_file_handle_to_log(log_step, filename, 'a', logging.WARNING)
    log = log_step

def _add_file_handle_to_log(logger, filename, mode, level):
    fileh = logging.FileHandler(filename, mode)
    fileh.setLevel(level)
    fileh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M'))
    logger.addHandler(fileh)
    logger.propagate = False
    return logger

"""
Pipeline variables and functions.
"""
import os, sys
import logging
import time
import numpy as np
from importlib import import_module
from collections import OrderedDict
from . import utils
from . import hrjson as json

# ------------------------------------------------------------------------------
# Pipeline paramters are all set during call of dsb3.init_pipeline(...)
# ------------------------------------------------------------------------------

avail_dataset_names = ['LUNA16', 'dsb3']

avail_steps = OrderedDict([
    ('step0', 'resample_lungs'),
    ('step1', 'gen_prob_maps'),
    ('step2', 'gen_candidates'),
    ('step4', 'gen_nodule_masks'),
])

avail_runs = OrderedDict([])
"""Stores optimization runs. Is read from file at startup."""

dataset_name = None
"""Either LUNA16 or dsb3."""

write_basedir = None
"""Toplevel directory to store all runs and datasets.""" 

n_patients = None
"""Number of patients."""

patients = None
"""List of all patients. Use this to iterate over patients."""

raw_data_dir = None
"""Directory with raw data."""

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

log_tf = None
"""Logfile for all the tensorflow unstructured C output."""

# technical parameters
n_CPUs = 1
"""Number of CPUs to use in multi-threading."""

GPU_ids = None
"""List of GPU ids for computations."""

GPU_memory_fraction = 0.85
"""Fraction of memory attributed to GPU computation."""

# track pipeline runs
_step_name = None
"""Stores which step is currently processed."""

__run = 0
"""Integer that identifies the current run of the pipeline."""

__init_run = 0
"""Integer that identifies the run that is used to initialize the current run."""

# ------------------------------------------------------------------------------
# User functions
# ------------------------------------------------------------------------------

def save_step(d, step_name=None):
    step_dir_ = step_dir if step_name is None else get_step_dir(step_name)
    json.dump(d, open(step_dir_ + 'out.json', 'w'), indent=4, indent_to_level=1)
    print('wrote', step_dir_ + 'out.json')

def load_step(step_name=None):
    step_dir = _get_step_dir_for_load(step_name)
    return json.load(open(step_dir + 'out.json'))

def save_array(basename, array, step_name=None):
    step_dir = get_step_dir(step_name) + 'arrays/'
    np.save(step_dir + basename, array)
    return step_dir + basename

def load_array(basename, step_name=None):
    step_dir = _get_step_dir_for_load(step_name) + 'arrays/'
    return np.load(step_dir + basename)

def get_write_dir(run=None):
    """Output directory where all processed data of a run is written."""
    run = __run if run is None else run
    return write_basedir  + dataset_name + '_' + str(run)  + '/'

def get_step_dir(step_name=None, run=None):
    """Output directory where all processed data of a specifc step in a specific run is written.
    Is subdirectory of `write_dir`."""
    step_name = _step_name if step_name is None else step_name
    return get_write_dir(run) + step_name + '/'

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def _get_step_dir_for_load(step_name=None):
    """
    Go backwards in run history to find the directory.
    """
    trial_runs = [__run] + list(range(__init_run, -1, -1))
    for run in trial_runs:
        step_dir = get_step_dir(step_name, run)
        if os.path.exists(step_dir):
            return step_dir
    raise FileNotFoundError('Did not find ' + step_dir + ' in runs ' + str(trial_runs) + '.')

def _init_run(next_step_name, run=-1, descr='', init_run=-1):
    runs_filename = write_basedir + dataset_name + '_runs.json'
    # case run == -1 and the step has already been run before
    global avail_runs
    if run == -1:
        # there have already been previous runs
        if os.path.exists(runs_filename):
            avail_runs = json.load(open(runs_filename), object_pairs_hook=OrderedDict)
            run = int(next(reversed(avail_runs)))
            if descr != '':
                print('increasing run level to', run + 1)
                run += 1 # increase run level by one
        else:
            run = 0
            descr = 'first_run' if descr == '' else descr
    # case run > -1: simply update avail_runs
    else:
        avail_runs = json.load(open(runs_filename), object_pairs_hook=OrderedDict)
    if avail_runs and descr == '':
        descr = avail_runs[str(run)][1]
    avail_runs[str(run)] = [time.strftime('%Y-%m-%d %H:%M', time.localtime()), descr]
    json.dump(avail_runs, open(runs_filename, 'w'), indent=4, indent_to_level=0)
    # update global variables
    global __run, __init_run, write_dir
    __run = run
    if init_run > -1:
        __init_run = init_run
    else:
        __init_run = run - 1
    # create directory if it's not there already
    utils.ensure_dir(get_write_dir())
    # logger for the whole pipeline, to be invoked by `pipe.log.info(msg)`, for example
    _init_log()

def _run_step(step_name, params):
    step_key = [k for (k, v) in avail_steps.items() if v == step_name][0]
    info = 'run ' + str(__run) + ' (' + avail_runs[str(__run)][1] + ')' + ' / ' + step_key + ' (' + step_name + ')' \
           + (' with init ' + str(__init_run) if __init_run > -1 else '')
    log_pipe.info(info)
    # output params dict for visual check
    print(len(info) * '_' + '\n' + info)
    for key, value in params.items():
        print('    {} = {}'.format(key, value))
    # create step directories, log files etc.
    global _step_name
    _step_name = step_name
    step_dir = get_step_dir()
    # data directory of step
    utils.ensure_dir(step_dir + 'arrays/')
    utils.ensure_dir(step_dir + 'figs/')
    # init step logger
    _init_log_step(step_name)
    log.info('start step with ' + ('init ' + str(__init_run)) if __init_run > -1 else 'default init (most recent run)')
    # saving params dict
    json.dump(params, open(step_dir + 'params.json', 'w'), indent=4, indent_to_level=0)
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
    finish_msg = '... finished the step'
    log.info(finish_msg)
    log_pipe.info(finish_msg)

def _init_patients():
    from collections import OrderedDict
    filename = get_write_dir() + 'patients_raw_data_paths.json'
    global patients_raw_data_paths
    global patients
    if os.path.exists(filename):
        patients_raw_data_paths = json.load(open(filename), object_pairs_hook=OrderedDict)
        patients = list(patients_raw_data_paths.keys())
    else:
        from glob import glob
        from natsort import natsorted
        if dataset_name == 'LUNA16':
            patient_paths = glob(raw_data_dir + 'subset*/*.mhd')
        elif dataset_name == 'dsb3':
            patient_paths = glob(raw_data_dir + '*/')
        patient_paths = natsorted(patient_paths, key=lambda p: p.split('/')[-1])
        if dataset_name == 'LUNA16':
            patients = [p.split('/')[-1].split('.')[-2] for p in patient_paths]
        elif dataset_name == 'dsb3':
            patients = [p.split('/')[-2] for p in patient_paths]
        patients_raw_data_paths = OrderedDict(zip(patients, patient_paths))
        utils.ensure_dir(filename)
        json.dump(patients_raw_data_paths, open(filename, 'w'), indent=4)
    global n_patients
    if n_patients > 0:
        patients = patients[:n_patients]
        patients_raw_data_paths = OrderedDict(list(patients_raw_data_paths.items())[:n_patients])
    else:
        n_patients = len(patients)
    print('processing', n_patients, 'patient', 's' if n_patients > 1 else '')

def _init_log(level=logging.INFO):
    global log_pipe
    filename = get_write_dir()+ 'log.txt'
    if os.path.exists(filename):
        open(filename, 'a').write('\n')
    log_pipe = logging.getLogger(filename)
    log_pipe.setLevel(level) # it's necessary to set the level also here
    log_pipe = _add_file_handle_to_log(log_pipe, filename, 'a', level)
    log_pipe.propagate = False

def _init_log_step(step_name, level=logging.INFO):
    global log_step, log
    step_dir = get_step_dir(step_name)
    filename = step_dir + 'log.txt'
    log_step = logging.getLogger(filename)
    log_step.setLevel(level) # it's necessary to set the level also here
    log_step = _add_file_handle_to_log(log_step, filename, 'w', level, passed_time=True)
    # write errors also to pipeline log file
    filename = get_write_dir()+ 'log.txt'
    log_step = _add_file_handle_to_log(log_step, filename, 'a', logging.WARNING)
    # write everthing also to stdout
    ch = logging.StreamHandler()
    ch.setFormatter(LogFormatter(passed_time=True))
    log_step.addHandler(ch)
    # update abbreviation
    log = log_step
    # tensorflow log file
    global log_tf
    log_tf = step_dir + 'log_tf.txt'
    open(log_tf, 'w').close()

class LogFormatter(logging.Formatter):
    msg  = '%(msg)s'
    def __init__(self, fmt='%(levelno)s: %(msg)s', datefmt='%Y-%m-%d %H:%M', style='%', passed_time=False):
        super().__init__(fmt, datefmt, style)
        self.passed_time = passed_time
        self.last_time = time.process_time()
    def format(self, record):
        format_orig = self._style._fmt
        if record.levelno == logging.INFO:
            current_time = time.process_time()
            passed_time_str = time.strftime('%H:%M:%S', time.gmtime(current_time - self.last_time))
            if self.passed_time:
                self._style._fmt = passed_time_str + ' - %(msg)s'
            else:
                self._style._fmt = '%(asctime)s | ' + passed_time_str +  ' - %(msg)s'
            self.last_time = time.process_time()
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result

def _add_file_handle_to_log(logger, filename, mode, level, passed_time=False):
    fileh = logging.FileHandler(filename, mode)
    fileh.setLevel(level)
    fileh.setFormatter(LogFormatter(passed_time=passed_time))
    logger.addHandler(fileh)
    return logger

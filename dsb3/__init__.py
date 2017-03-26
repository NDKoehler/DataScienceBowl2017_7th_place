import os
import numpy as np
from collections import OrderedDict
from . import utils

def init_pipeline(run,
                  run_descr,
                  single_patient_id,
                  fromto_patients,
                  n_patients,
                  dataset_name,
                  raw_data_dirs,
                  write_basedir,
                  tr_va_ho_split=None,
                  random_seed=17,
                  n_CPUs=1,
                  GPU_ids=None,
                  GPU_memory_fraction=0.85):
    """
    Initialize pipeline.
    """
    from . import pipeline as pipe
    # checks
    if dataset_name not in pipe.avail_dataset_names:
        raise ValueError('dataset_name needs to be one of' + str(pipe.avail_dataset_names))
    print('processing dataset', dataset_name)
    # set a pipeline attributes
    pipe.dataset_name = dataset_name # the dataset identifier
    pipe.raw_data_dir = raw_data_dirs[dataset_name] # the raw data
    # create base directory
    pipe.write_basedir = write_basedir.rstrip('/') + '/'
    utils.ensure_dir(pipe.get_write_dir())
    # init the current run of the pipeline
    pipe._init_run(run, run_descr)
    # logger for the whole pipeline, to be invoked by `pipe.log.info(...)`
    pipe._init_log_pipe()
    # set seed
    np.random.seed(random_seed)
    # locate input data files and init patient lists
    pipe._init_patients(n_patients, single_patient_id, fromto_patients)
    if pipe._init_patients_by_label():
        pipe._init_patients_by_split(tr_va_ho_split)
    # technical parameters
    pipe.n_CPUs = n_CPUs
    pipe.GPU_memory_fraction = GPU_memory_fraction
    pipe.GPU_ids = GPU_ids
    print('with GPUs_ids', GPU_ids)
    # --------------------------------------------------------------------------
    # environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '100' # disable tensorflow info and warning logs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in GPU_ids])

import os
import numpy as np
from collections import OrderedDict

def init_pipeline(dataset_name,
                  raw_data_dirs,
                  write_basedir,
                  n_patients=0,
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
    if not os.path.exists(write_basedir):
        os.makedirs(write_basedir)
    pipe.write_basedir = write_basedir.rstrip('/') + '/'
    pipe.n_patients = n_patients
    # logger for the whole pipeline, to be invoked by `pipe.log.info(msg)`, for example
    pipe._init_log()
    # locate input data files and init patient lists
    np.random.seed(random_seed)
    pipe._init_patients()
    # pipe._init_patients_by_label()
    # pipe._init_patients_by_split(tr_va_ho_split)
    # technical parameters
    pipe.n_CPUs = n_CPUs
    pipe.GPU_memory_fraction = GPU_memory_fraction
    pipe.GPU_ids = GPU_ids
    print('with GPUs_ids', GPU_ids)
    # --------------------------------------------------------------------------
    # environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '100' # disable tensorflow info and warning logs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in GPU_ids])

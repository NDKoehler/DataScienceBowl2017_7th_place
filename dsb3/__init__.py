import os
import numpy as np

def init_pipeline(dataset_name,
                  dataset_dirs,
                  write_basedir,
                  random_seed=17,
                  tr_va_holdout_fractions=None,
                  n_CPUs=1,
                  GPU_ids=None,
                  GPU_memory_fraction=0.85,
                  n_patients_to_process=0):
    """
    Initialize pipeline.
    """
    from . import pipeline as pipe
    if dataset_name not in pipe.avail_dataset_names:
        raise ValueError('dataset_name needs to be one of' + str(avail_dataset_names))
    pipe.dataset_name = dataset_name
    pipe.dataset_dir = dataset_dirs[dataset_name]
    pipe.write_dir = write_basedir.rstrip('/') + '/' + dataset_name + '/'
    # locate input data files and init patient lists
    pipe.init_patients(n_patients_to_process)
    # logger for the whole pipeline, to be invoked by `pipe.log.info(msg)`, for example
    pipe.init_log()
    # technical parameters
    pipe.n_CPUs = n_CPUs
    pipe.GPU_memory_fraction = GPU_memory_fraction
    pipe.GPU_ids = GPU_ids
    np.random.seed(random_seed)
    # --------------------------------------------------------------------------
    # environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tensorflow info and warning logs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in GPU_ids])

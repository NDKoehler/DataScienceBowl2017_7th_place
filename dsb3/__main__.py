"""
The DSB3 Pipeline.

This is the general-purpose command-line utility.
"""

import os, sys, shutil
import argparse
import logging
import getpass
from collections import OrderedDict
from . import init_pipeline
from . import utils
from . import pipeline as pipe

def main():
    # --------------------------------------------------------------------------
    # import the master_config
    user_first_name = getpass.getuser().split('.')[0].split('_')[0]
    params_user = 'params_' + user_first_name + '.py'
    if not os.path.exists(params_user):
        raise FileNotFoundError('Provide file ' + params_user
                                + ' in the root of the repository.')
    shutil.copyfile(params_user, 'dsb3/params.py')
    from . import params
    # --------------------------------------------------------------------------
    # command line parameters
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=steps_descr()+runs_descr(), add_help=False)
    aa = parser.add_argument_group('Choose "steps".').add_argument
    aa('steps', type=str,
       help='Step key or name, for example, "0,1,2" or "foo_name,bar_name". See the choices below.')
    aa = parser.add_argument_group('Manage pipeline "runs", i.e., repeated sequences of "steps"').add_argument
    aa('-r', '--run', type=int, default=-1, metavar='i',
       help='Run key (default: work within current run).')
    aa('-d', '--descr', type=str, default='', metavar='d',
       help='Trigger a new run by providing a description for it.')
    aa('--init_run', type=int, default=-1, metavar='i',
       help='Select the initialization run (default: the previous run).')
    aa = parser.add_argument_group('Other').add_argument
    aa('-h', '--help', action='help',
       help='Show this help message and exit.')
    aa('-n', '--n_patients', type=int, default=None, metavar='n',
       help='Choose the number of patients to process to test the pipeline (default: read from params file).')
    aa('patient', nargs='?', default=None,
       help="provide patient id")
    aa('-ds', '--dataset_name', type=str, default=None, metavar='d',
       help='Choose dataset_name "dsb3" (default: read from params file).')
    aa('--gpu', type=str, default=None, metavar='gpu',
       help='Choose GPU_id, e.g., 0 (default: read from params file).')
    args = parser.parse_args()
    if ',' in args.steps:
        step_names = args.steps.split(',')
    else:
        step_names = [args.steps]
    # --------------------------------------------------------------------------
    # some output and checks
    for istep, step_name in enumerate(step_names):
        if step_name not in pipe.avail_steps and step_name not in set(pipe.avail_steps.values()):
            raise ValueError('Choose step to be one or a combination of\n' + steps_descr())
        if step_name in pipe.avail_steps:
            step_name = pipe.avail_steps[step_name]
            step_names[istep] = step_name
        if not hasattr(params, step_name):
            raise ValueError('Provide a parameter dict for step '
                             + step_name + ' in params_ ' + user_first_name + '!')
        if not os.path.exists('./dsb3/steps/' + step_name + '.py'):
            raise ValueError('Do not know any step called ' + step_name +'.')
    # --------------------------------------------------------------------------
    # overwrite default parameters
    if args.n_patients is not None:
        params.pipe['n_patients'] = args.n_patients
    if args.dataset_name is not None:
        dataset_name = args.dataset_name
        params.pipe['dataset_name'] = dataset_name
    if args.gpu is not None:
        GPU_id = args.gpu
        params.pipe['GPU_ids'] = [int(GPU_id)]
    # --------------------------------------------------------------------------
    # init pipeline
    init_pipeline(**params.pipe)
    # reinit the patient iterator list in pipe
    if args.patient is not None:
        pipe.patient = [patient]
    # now we can import `pipeline` from anywhere and use its attributes
    # --> plays the role of a class with only a single instance across the module
    # init the current run of the pipeline
    pipe._init_run(step_names[0], args.run, args.descr)
    for step_name in step_names:
        pipe._run_step(step_name, getattr(params, step_name))
        #pipe._visualize_step(step_name)

def steps_descr():
    descr = 'Choices for "steps":'
    for key, value in pipe.avail_steps.items():
        descr += '\n  {:<6} '.format(key) + value
    return descr

def runs_descr():
    import json
    from . import params
    # the following is a hack to get help output for argparse
    runs_filename = params.pipe['write_basedir'].rstrip('/') + '/' + params.pipe['dataset_name'] + '_runs.json'
    descr = '\n\nChoices for "--run" and "--init_run":'
    if os.path.exists(runs_filename):
        runs_dict = json.load(open(runs_filename), object_pairs_hook=OrderedDict)
        for key, value in runs_dict.items():
            descr += '\n  {:3} {} : {}'.format(key, value[0], value[1])
    else:
        descr += '\n  no runs performed yet'
    return descr

"""
The DSB3 Pipeline

This is the general-purpose command-line utility.
"""

import os, sys, shutil
import argparse
import logging
import getpass
from collections import OrderedDict
from . import init_pipeline
from . import utils

steps = OrderedDict([
    ('step0', 'resample_lungs'),
    ('step1', 'gen_prob_maps'),
    ('step2', 'gen_candidates'),
])

def main():
    # --------------------------------------------------------------------------
    # import the master_config
    user_first_name = getpass.getuser().split('.')[0].split('_')[0]
    params_user = 'params_' + user_first_name + '.py'
    if not os.path.exists(params_user):
        raise FileNotFoundError('Provide file ' 
                                + params_user 
                                + ' in the root of the repository.')
    shutil.copyfile(params_user, 'dsb3/params.py')
    print('... copied', params_user, 'to', 'dsb3/params.py')
    from . import params
    # --------------------------------------------------------------------------
    # command line parameters
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=steps_descr())
    aa = parser.add_argument
    aa('step', type=str, 
       help='One of the choices below, either as "stepX" or "step_name" '
            'or concatenated, for example, "step0,step1,step2" or "foo_name,bar_name".')
    aa('-a', '--action', choices=['run', 'vis', 'all'], default='run',
       help='"run" runs the step making predictions. '
            '"vis" visualizes the results of the step. '
            '"all" does everything. (default: "run").')
    aa('-n', '--n_patients_to_process', type=int, default=0,
       help='Choose the number of patients to process, to test the pipeline (default: process all patients).')
    args = parser.parse_args()
    if ',' in args.step:
        step_names = args.step.split(',')
    else:
        step_names = [args.step]
    # --------------------------------------------------------------------------
    # some checks
    for istep, step_name in enumerate(step_names):
        if step_name not in steps and step_name not in set(steps.values):
            raise ValueError()
        if step_name in steps:
            step_name = steps[step_name]
            step_names[istep] = step_name
        if not hasattr(params, step_name):
            raise ValueError('Provide a parameter dict for step '
                             + step_name + ' in params_ ' + user_first_name + '!')
        if not os.path.exists('./dsb3/steps/' + step_name + '.py'):
            raise ValueError('Do not know any step called ' + step_name +'.')
    # --------------------------------------------------------------------------
    # overwrite default params
    if args.n_patients_to_process > 0:
        params.pipe['n_patients_to_process'] = args.n_patients_to_process
    # --------------------------------------------------------------------------
    # init pipeline
    init_pipeline(**params.pipe)
    # now we can import `pipeline` from anywhere and use its attributes
    # --> plays the role of a class with only a single instance across the module
    from . import pipeline as pipe
    # perform action
    for step_name in step_names:
        if args.action == 'run':
            pipe.run_step(step_name, getattr(params, step_name))
        elif args.action == 'vis' or args.action == 'all':
            pipe.vis_step(step_name)

def steps_descr():
    descr = 'choices for step:'
    for key, value in steps.items():
        descr += '\n  {:12}'.format(key) + value
    return descr

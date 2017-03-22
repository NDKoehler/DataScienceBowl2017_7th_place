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
    aa = parser.add_argument_group('Choose "steps" and the actions to take.').add_argument
    aa('steps', type=str, 
       help='See the choices below, for example, "step0,step1,step2" or "foo_name".')
    aa('-a', '--action', choices=['go', 'vis', 'all'], default='go', metavar='a',
       help='"go": run steps and make predictions, '
            '"vis": visualize the results, '
            '"all": do both (default: "go").')
    aa = parser.add_argument_group('Manage pipeline "runs", i.e., repeated sequences of "steps"').add_argument
    aa('-r', '--run', type=int, default=-1, metavar='i',
       help='Select a "run" (default: work within current run, only start a new run if "-rd" is provided).')
    aa('-d', '--descr', type=str, default='', metavar='d',
       help='Trigger a new run by providing a description for it.')
    aa('--init_run', type=int, default=-1, metavar='i',
       help='Select a "run" for initialization (default: the previous run).')
    aa = parser.add_argument_group('Other').add_argument
    aa('-h', '--help', action='help',
       help='Show this help message and exit.')
    aa('-n', '--n_patients', type=int, default=0, metavar='n',
       help='Choose the number of patients to process to test the pipeline (default: process all patients).')
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
    # overwrite default params
    if args.n_patients > 0:
        params.pipe['n_patients'] = args.n_patients
    # --------------------------------------------------------------------------
    # init pipeline
    init_pipeline(**params.pipe)
    # now we can import `pipeline` from anywhere and use its attributes
    # --> plays the role of a class with only a single instance across the module
    # init the current run of the pipeline
    pipe._init_run(step_names[0], args.run, args.descr)
    # perform action
    for step_name in step_names:
        if args.action == 'go':
            pipe._run_step(step_name, getattr(params, step_name))
        elif args.action == 'vis' or args.action == 'all':
            raise ValueError('not implemented yet')
            pipe._vis_step(step_name)

def steps_descr():
    descr = 'Choices for "steps":'
    for key, value in pipe.avail_steps.items():
        descr += '\n  {:12}'.format(key) + value
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

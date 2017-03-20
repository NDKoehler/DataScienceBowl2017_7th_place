import os
import json
import logging

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        os.system('chmod -R ugo+rw ' + d)

def get_logger(filename, mode='w', level=logging.INFO):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    fileh = logging.FileHandler(filename, mode)
    fileh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M'))
    logger.addHandler(fileh)
    logger.propagate = False
    return logger

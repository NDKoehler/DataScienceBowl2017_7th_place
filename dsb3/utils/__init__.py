import os
import json
import logging

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        os.system('chmod -R ugo+rw ' + d)

def dir_is_empty(directory):
    for _, _, files in os.walk(directory):
        if files:
            return False
        else:
            return True

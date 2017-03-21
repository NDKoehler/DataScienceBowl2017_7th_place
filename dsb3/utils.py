import os
import json
import logging

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        os.system('chmod -R ugo+rw ' + d)

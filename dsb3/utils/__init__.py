import os
import json
import logging
import numpy as np

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

def crop_and_embed(array, box_coords, cube_shape):
    """Take box in array and match into cube_shape. Return the cube_shape array.
    box_coords : is of length 6
    cube_shape : is of length 3
    """
    cube_array = np.zeros(cube_shape, dtype=array.dtype)
    crop = [int(max(0, box_coords[i])) 
            if i % 2 == 0 else
            int(min(box_coords[i], array.shape[i // 2]))
            for i in range(6)]
    embd = [abs(box_coords[i]) if box_coords[i] < 0 else 0
            if i % 2 == 0 else
            cube_shape[i // 2] if box_coords[i] <= array.shape[i // 2] else array.shape[i // 2] - box_coords[i]
            for i in range(6)]
    cube_array[embd[0]:embd[1], embd[2]:embd[3], embd[4]:embd[5]] \
               = array[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]
    return cube_array

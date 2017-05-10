import json
import pandas as pd
import sys, os
from collections import OrderedDict


dirs = ['LUNA16', 'dsb3']
sets = ['tr', 'va', 'ho']

for dir in dirs:
    patients_by_split = OrderedDict()


    for split in sets:

        spath = './' + dir + '/list/' + dir + '_' + split + '.lst'
        if os.path.exists(spath):
            try:
                lst = pd.read_csv(spath, header=None, sep='\t')[0].values.tolist()
            except:
                print(spath, "is empty")
                lst = [] #empty list

        else:
            print("list missing:", spath)
            sys.exit()
        
        patients_by_split[split] = lst
    
    filename = './' + dir + '/json/' + 'patients_by_split.json'
    json.dump(patients_by_split, open(filename, 'w'), indent=4)
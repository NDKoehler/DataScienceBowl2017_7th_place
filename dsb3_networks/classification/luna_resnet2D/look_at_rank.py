import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys, os
import seaborn as sns

def load_json(filename):
    with open(filename) as f:
        d = json.load(f)
    return d

net = sys.argv[1]
filename = net + '/rank_log.json'

ranks = load_json(filename)
rank_cluster = load_json('filter_rank.json')['0']



print(ranks.keys())


rank = ranks[str(len(ranks.keys())-1)]
ids = np.nonzero(np.array(rank))[0]
ids_cluster = np.nonzero(np.array(rank_cluster))[0]


print(len(ids), len(ids_cluster))

boundary=1000

ids[ids > boundary] = boundary
ids_cluster[ids_cluster > boundary] = boundary
sns.distplot(ids, kde=False, color='blue', label="network", bins=50)
sns.distplot(ids_cluster, kde=False, color='red', label="cluster", bins=50)

plt.ylim(0,40)
plt.legend()
plt.show()




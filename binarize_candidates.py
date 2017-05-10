import numpy as np
import pandas as pd


tr = pd.read_csv('../datapipeline_final/LUNA16_0/interpolate_candidates/tr_candidates.lst',header=None, sep = '\t')
va = pd.read_csv('../datapipeline_final/LUNA16_0/interpolate_candidates/va_candidates.lst',header=None, sep = '\t')


print(tr)

tr[1][tr[1] <= 2] = 0
va[1][va[1] <= 2] = 0
tr[1][tr[1] > 2] = 1
va[1][va[1] > 2] = 1

tr.to_csv('../datapipeline_final/LUNA16_0/interpolate_candidates/tr_candidates_binary.lst', header=None, sep = '\t', index = False)
va.to_csv('../datapipeline_final/LUNA16_0/interpolate_candidates/va_candidates_binary.lst', header=None, sep = '\t', index = False)


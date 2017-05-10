import glob
import numpy as np
import sys, os

np.random.seed(17) # do NOT change

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

if __name__ == '__main__':
    tr_va_ratio = 0.9 # do not change

    all_masks = np.random.permutation(glob.glob('../data/annotations/lung_wings_imgs/*_mask.jpg'))
    ensure_dir('../data/lsts/')

    tr_len = int(tr_va_ratio*len(all_masks))

    tr_lst = list(all_masks[:tr_len])
    tr_lst = np.random.permutation(tr_lst)
    va_lst = np.random.permutation(list(all_masks[tr_len:]))

    if len(set(tr_lst).intersection(set(va_lst)))>0:
        print('ERROR same objects {} in tr und va lst'.format(set(tr_lst).intersection(set(va_lst))) )
        sys.exit()

    with open('../data/lsts/tr.lst', 'w') as lst:
        for cnt,mask_path in enumerate(tr_lst):
            img_path = mask_path.split('_mask')[-2]+'.jpg'
            lst.write('{}\t{}\t{}\n'.format(cnt,os.path.abspath(mask_path),os.path.abspath(img_path)))

    with open('../data/lsts/va.lst', 'w') as lst:
        for cnt,mask_path in enumerate(va_lst):
            img_path = mask_path.split('_mask')[-2]+'.jpg'
            lst.write('{}\t{}\t{}\n'.format(cnt,os.path.abspath(mask_path),os.path.abspath(img_path)))

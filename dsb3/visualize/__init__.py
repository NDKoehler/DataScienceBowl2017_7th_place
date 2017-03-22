import os
import grip
from collections import OrderedDict

def write_figs_overview_html(figs_directory, show_image_info=True):
    sorted_filenames_dict = get_filenames_sorted_by_the_last_element_of_basename(figs_directory)
    n_rows = max([len(val) for val in sorted_filenames_dict.values()])
    with open(figs_directory.rstrip('/') + '.md', 'w') as f:
        write_col_names(f, sorted_filenames_dict.keys())
        for irow in range(n_rows):
            if show_image_info:
                write_col_img_info(f, irow, sorted_filenames_dict)
            write_col_img_inludes(f, irow, sorted_filenames_dict)
    grip.export(figs_directory.rstrip('/') + '.md')
    os.remove(figs_directory.rstrip('/') + '.md')

def write_col_img_info(f, row_idx, sorted_filenames_dict):
    f.write('| ')
    for filenames in sorted_filenames_dict.values():
        if row_idx < len(filenames):
            f.write(get_cropped_basename_without_ext_and_last_element(filenames[row_idx]))
        f.write(' | ')
    f.write('\n')

def write_col_img_inludes(f, row_idx, sorted_filenames_dict):
    f.write('| ')
    for filenames in sorted_filenames_dict.values():
        if row_idx < len(filenames):
            f.write('<img src="figs/' + filenames[row_idx] + '" style="width: 130px"> ')
        f.write(' | ')
    f.write(' | ')
    f.write('\n')

def write_col_names(f, col_names):
    f.write('| ')
    for name in col_names:
        f.write(name + ' | ')
    f.write('\n|')
    for _ in col_names:
        f.write(':-:|')
    f.write('\n')

def get_cropped_basename_without_ext_and_last_element(filename):
    name = os.path.basename(filename).split('.')[0].split('_')[0]
    if len(name) > 30:
        name = name[:12] + '...' + name[-12:]
    return name

def get_filenames_sorted_by_the_last_element_of_basename(directory):
    """
    No filenames containing dots other than for the filetype ending are allowed.
    """
    from natsort import natsorted
    filenames = natsorted(os.listdir(directory))
    sorted_filenames_dict = OrderedDict([])
    for filename in filenames:
        label = filename.split('.')[0].split('_')[-1]
        if label in sorted_filenames_dict:
            sorted_filenames_dict[label].append(filename)
        else:
            sorted_filenames_dict[label] = [filename]
    return sorted_filenames_dict

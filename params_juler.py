from collections import OrderedDict

# ------------------------------------------------------------------------------
# pipeline parameters
# ------------------------------------------------------------------------------

pipe = OrderedDict([
    ('n_patients', 0), # number of patients to process, 0 means all
# dataset origin and paths
    ('dataset_name', 'LUNA16'), # 'LUNA16' or 'dsb3'
    ('raw_data_dirs', {
        'LUNA16': '/media/juler/Data_0/dsb3/raw_data/LUNA16/',
        'dsb3': '/media/juler/Data_0/dsb3/raw_data/dsb3/stage1/',
        #'dsb3': '/',
    }),
    #('write_basedir', '/media/juler/qnap/PROJECTS/dsb3/data_pipeline_gen2/'),
    #('write_basedir', '/media/juler/Data_0/dsb3/datapipelines/dsb3/'),
    ('write_basedir', '/media/juler/Data_0/dsb3/datapipelines/LUNA16_0/'),
    #('write_basedir', '/home/juler/Projects/dsb3a/test/'),
# data splits
    ('random_seed', 17),
                       # tr  va   ho
    ('tr_va_ho_split', [0.8, 0.2, 0]), # something like 0.15, 0.7, 0.15
# technical parameters
    ('n_CPUs', 7),
    ('GPU_ids', [0]),
    ('GPU_memory_fraction', 0.85),
])

# ------------------------------------------------------------------------------
# step parameters
# ------------------------------------------------------------------------------
all_patients = False
assets_path = '../dsb3a_assets/'

resample_lungs = OrderedDict([
    ('new_spacing_zyx', [1, 1, 1]), # z, y, x
    ('HU_tissue_range', [-1000, 400]), # MIN_BOUND, MAX_BOUND [-1000, 400]
    ('data_type', 'int16'), # int16 or float32
    ('bounding_box_buffer_yx_px', [12, 12]), # y, x
    ('seg_max_shape_yx', [512, 512]), # y, x
    ('batch_size', 64), # 128 for target_spacing 0.5, 64 for target_spacing 1.0
    ('checkpoint_dir', assets_path+'checkpoints/resample_lungs/lung_wings_segmentation'),
])

gen_prob_maps = OrderedDict([
    # the following two parameters are critical for computation time and can be easily changed
    ('view_planes', 'zyx'), # a string consisting of 'y', 'x', 'z'
    ('view_angles', [0]), # per view_plane in degrees, for example, [0, 45, -45]
    # more technical parameters
    ('image_shapes', [[304, 304], [320, 320], [352, 352], [384, 384], [400, 400], [416, 416],
                     [432, 432], [448, 448], [480, 480], [512, 512], [560, 560], [1024, 1024]]), # y, x
                     # valid shape numbers: 256, 304, 320, 352, 384, 400, 416, 448, 464, 480, 496, 512 (dividable by 16)
    ('batch_sizes',  [32, 32, 24, 24, 16, 16, 16, 16, 12, 12, 4, 1]),
    ('data_type', 'uint8'), # uint8, int16 or float32
    ('image_shape_max_ratio', 0.98),
    ('checkpoint_dir', assets_path+'checkpoints/gen_prob_maps/ellips_05reduced_rad_multiIDX_logloss_1classWeight_sliceCompensation_stagedIntensity'),
    ('all_patients', all_patients),
])
gen_candidates = OrderedDict([
    ('n_candidates', 100),
    ('ensemble_foldername_of_prob_maps', ['gen_prob_maps_logloss','gen_prob_maps_logloss_stagedInten']), # False=gen_prob_maps else list of foldernames in datapipeline_directory
    ('threshold_prob_map', 0.1),
    ('cube_shape', (32, 32, 32)), # ensure cube_edges are dividable by two -> improvement possible
    ('all_patients', all_patients),
])

interpolate_candidates = OrderedDict([
    ('n_candidates', 100),
    ('new_spacing_zyx', [0.5, 0.5, 0.5]), # y, x, z
    ('new_data_type', 'uint8'),
    ('new_candidates_shape_zyx', [64, 64, 64]),
    ('crop_raw_scan_buffer', 10), # hard code
])

filter_candidates = OrderedDict([
    ('n_candidates', 20),
    ('checkpoint_dir', assets_path+'checkpoints/filter_candidates/rank_cross3/'),
    ('num_augs_per_img',1 ), # 1 means NO augmentation	
    ('all_patients', all_patients),
])

gen_submission = OrderedDict([
    ('splitting', 'validation'), # 'validation' or 'submission' or 'holdout'<-not implemented yet
    ('checkpoint_dir', assets_path+'checkpoints/gen_submission/cross_crop_retrain_further'),
    ('patients_lst_path', 'filter_candidates'), # if False->filter_candidates; if 'interpolate_candidates'->lst and arrays from current dataset is used
    ('num_augs_per_img', 5),
    ('sample_submission_lst_path', '../raw_data/dsb3/stage1_sample_submission.csv'),
])

# ------------------------------------------------------------------------------
# nodule segmentation parameters
# ------------------------------------------------------------------------------

gen_nodule_masks = OrderedDict([
    ('ellipse_mode', True),
    ('reduced_mask_radius_fraction', 0.5),
    ('mask2pred_lower_radius_limit_px', 5),
    ('mask2pred_upper_radius_limit_px', 20),
    ('LUNA16_annotations_csv_path', assets_path+'LIDC-annotations_2_nodule-seg_annotations/annotations_min+missing_LUNA16_patients.csv'),
    ('yx_buffer_px', 0),
    ('z_buffer_px', 0),
])

gen_nodule_seg_data = OrderedDict([
    ('view_angles', [0]), # per view_plane (degree)
    ('extra_radius_buffer_px', 15),
    ('num_channels', 1),
    ('stride', 1),
    ('crop_size', [96, 96]),
    ('view_planes', 'yxz'), 
    ('num_negative_examples_per_nodule_free_patient_per_view_plane', 40),
    ('HU_tissue_range', [-1000, 400]), # MIN_BOUND, MAX_BOUND [-1000, 400]
])


# ------------------------------------------------------------------------------
# Eval parameters
# ------------------------------------------------------------------------------

gen_candidates_eval = OrderedDict([
    ('max_n_candidates', 20),
    ('max_dist_fraction', 0.5),
    ('priority_threshold', 3), 
    ('sort_candidates_by', 'prob_sum_min_nodule_size'),#'prob_sum_min_nodule_size', 'nodule_score'),
    ('all_patients', False)
])

gen_candidates_vis = OrderedDict([
    ('inspect_what', 'true_positives')
])

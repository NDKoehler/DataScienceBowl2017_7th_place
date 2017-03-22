from collections import OrderedDict

# ------------------------------------------------------------------------------
# pipeline parameters
# ------------------------------------------------------------------------------

pipe = OrderedDict([
    ('n_patients', 1), # number of patients to process, 0 means all
# dataset origin and paths
    ('dataset_name', 'LUNA16'), # 'LUNA16' or 'dsb3'
    ('raw_data_dirs', {
        'LUNA16': '/home/alex_wolf/storage/dsb3/data_raw/LUNA16/',
        'dsb3': '/home/alex_wolf/storage/dsb3/data_raw/dsb3/stage1/',
    }),
    ('write_basedir', '/home/alex_wolf/test/'),
# data splits
    ('random_seed', 17),
    ('tr_va_ho_split', [0.2, 0.8, 0]), # something like 0.15, 0.7, 0.15
# technical parameters
    ('n_CPUs', 7),
    ('GPU_ids', [0]),
    ('GPU_memory_fraction', 0.85),
])

# ------------------------------------------------------------------------------
# step parameters
# ------------------------------------------------------------------------------

resample_lungs = OrderedDict([
    ('target_spacing_zyx', [1, 1, 1]), # z, y, x
    ('HU_tissue_range', [-1000, 400]), # MIN_BOUND, MAX_BOUND [-1000, 400]
    ('data_type', 'int16'), # int16 or float32
    ('bounding_box_buffer_yx_px', [12, 12]), # y, x
    ('seg_max_shape_yx', [512, 512]), # y, x
    ('batch_size', 64), # 128 for target_spacing 0.5, 64 for target_spacing 1.0

    ('checkpoint_dir', './checkpoints/resample_lungs/lung_wings_segmentation'),
])

gen_prob_maps = OrderedDict([
    # the following two parameters are critical for computation time and can be easily changed
    ('view_planes', 'z'), # a string consisting of 'y', 'x', 'z'
    ('view_angles', [0]), # per view_plane in degrees, for example, [0, 45, -45]
    # more technical parameters
    ('image_shapes', [[304, 304], [320, 320], [352, 352], [384, 384], [400, 400], [416, 416],
                     [432, 432], [448, 448], [480, 480], [512, 512], [560, 560], [1024, 1024]]), # y, x
                     # valid shape numbers: 256, 304, 320, 352, 384, 400, 416, 448, 464, 480, 496, 512 (dividable by 16)
    ('batch_sizes',  [32, 32, 24, 24, 16, 16, 16, 16, 12, 12, 4, 1]),
    ('data_type', 'uint8'), # uint8, int16 or float32
    ('image_shape_max_ratio', 0.95),
    ('checkpoint_dir', './checkpoints/gen_prob_maps/nodule_seg_1mm_128x128_5Channels_multiview'),
])

gen_candidates = OrderedDict([
    ('max_n_candidates_per_patient', 20),
    ('padding_candidates', True),
    ('threshold_prob_map', 0.2),
    ('cube_shape', (48, 48, 48)), # ensure cube_edges are dividable by two -> improvement possible
])

interpolate_candidates = OrderedDict([
    ('HR_target_spacing_yxz', [0.5, 0.5, 0.5]), # y, x, z
    ('in_candidates_folder_name', '2017_03_15-20_56'), # '2017_03_11-08_06'
    ('num_candidates', 20),
    ('out_folder_name', 'multiview-3'),
    ('out_datatype', 'uint8'),
    ('out_candidates_shape_zyx', [96, 96, 96]),
    ('crop_raw_scan_buffer', 10),
])

filter_candidates = OrderedDict([
    ('checkpoint_dir', './checkpoints/luna_candidate_level_mini/'),
])

gen_submission = OrderedDict([
    ('splitting', 'submission'), # 'validation' or 'submission' or 'holdout'
    ('checkpoint_dir', './checkpoints/test'),
    ('num_augmented_data', 15), # is batch size
    ('gpu_fraction', 0.85),
    ('is_training', True),
])

# ------------------------------------------------------------------------------
# nodule segmentation parameters
# ------------------------------------------------------------------------------

gen_nodule_masks = OrderedDict([
    ('reduced_mask_radius_fraction', 0.5),
    ('mask2pred_lower_radius_limit_px', 3),
    ('mask2pred_upper_radius_limit_px', 15),
    ('LUNA16_annotations_csv_path', '../dsb3a_assets/LIDC-annotations_2_nodule-seg_annotations/annotations_min+missing_LUNA16_patients.csv'),
    ('num_minimal_affected_layers', 5),
    ('yx_buffer_px', 1),
    ('z_buffer_px', 2),
])

gen_nodule_seg = OrderedDict([
    ('records_extra_radius_buffer_px', 5),
    ('gen_records_num_channels', 1),
    ('gen_records_stride', 1),
    ('gen_records_crop_size', [128, 128]), # y,x
    ('ratio_nodule_nodule_free', 1.0),
    ('view_planes', 'yxz'), # 'y' enables y-plane as nodule view, 'yx' x- and y-plane,... (order is variable)
    ('view_angles', [0, 45]), # per view_plane (degree)
    ('num_negative_examples_per_nodule_free_patient_per_view_plane', 50),
# gen nodule segmentation lists
    ('data_extra_radius_buffer_px', 5),
    ('data_num_channels', 1),
    ('data_stride', 1),
    ('data_crop_size', [96, 96]), # y, x
    ('data_view_planes', 'yxz'), # 'y' enables y-plane as nodule view, 'yx' x- and y-plane, ... (order is variable)
    ('num_negative_examples_per_nodule_free_patient_per_view_plane', 40),
])

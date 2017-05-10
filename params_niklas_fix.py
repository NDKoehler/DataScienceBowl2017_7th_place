from collections import OrderedDict

# ------------------------------------------------------------------------------
# pipeline parameters
# ------------------------------------------------------------------------------
raw_data_absolute_path = '/media/niklas/Storage/dsb3/dsb3_raw/'
raw_LUNA_absolute_path = '/media/niklas/Storage/dsb3/LUNA16/'


pipe = OrderedDict([
    ('n_patients', 0), # number of patients to process, 0 means all
# dataset origin and paths
    ('dataset_name', 'dsb3'), # 'LUNA16' or 'dsb3'
    ('raw_data_dirs', {
        'LUNA16': raw_LUNA_absolute_path,
        'dsb3': raw_data_absolute_path + 'stage?/',
    }),
    ('write_basedir', './datapipeline_final/'),
# data splits
    ('random_seed', 17),
    ('tr_va_ho_split', [0.8, 0.2, 0]), # something like 0.15, 0.7, 0.15
# technical parameters
    ('n_CPUs', 10),
    ('GPU_ids', [0]),
    ('GPU_memory_fraction', 0.85),
])
# -----------------------------------------------------------------------------
# step parameters
# ------------------------------------------------------------------------------
validate_seg_net_on_all_patients = True


resample_lungs = OrderedDict([
    ('new_spacing_zyx', [1, 1, 1]), # z, y, x
    ('HU_tissue_range', [-1000, 400]), # MIN_BOUND, MAX_BOUND [-1000, 400]
    ('data_type', 'int16'), # int16 or float32
    ('bounding_box_buffer_yx_px', [12, 12]), # y, x
    ('seg_max_shape_yx', [512, 512]), # y, x
    ('batch_size', 64), # 128 for new_spacing 0.5, 64 for new_spacing 1.0
    ('checkpoint_dir', './checkpoints/lung_wings_segmentation/'),
])
batch_size_factor = 1
gen_prob_maps = OrderedDict([
    # the following two parameters are critical for computation time and can be easily changed
    ('view_planes', 'zyx'), # a string consisting of 'y', 'x', 'z'
    ('view_angles', [0]), # per view_plane in degrees, for example, [0, 45, -45]
    # more technical parameters
    # valid shape numbers: 256, 304, 320, 352, 384, 400, 416, 448, 464, 480, 496, 512 (dividable by 16)
    ('image_shapes', [[304, 304], [320, 320], [352, 352], [384, 384], [400, 400], [416, 416],
                     [432, 432], [448, 448], [480, 480], [512, 512], [560, 560], [1024, 1024]]), # y, x
    ('batch_sizes',  [batch_size_factor*32, batch_size_factor*32, batch_size_factor*24, batch_size_factor*24, batch_size_factor*16, batch_size_factor*16, batch_size_factor*16, batch_size_factor*16, batch_size_factor*12, batch_size_factor*12, batch_size_factor*4, batch_size_factor*1]),
    ('data_type', 'uint8'), # uint8, int16 or float32
    ('image_shape_max_ratio', 0.95),
    ('checkpoint_dir', './checkpoints/nodule_segmentation/'),
    ('all_patients', validate_seg_net_on_all_patients)
])


gen_candidates = OrderedDict([
    ('n_candidates', 20), #10
    ('sort_clusters_by', 'prob_sum_min_nodule_size'), #'prob_sum_min_nodule_size' #prob_sum_min_nodule_size # prob_sum_cluster
    ('threshold_prob_map', 0.2),
    ('cube_shape', (32, 32, 32)), # ensure cube_edges are dividable by two -> improvement possible
    ('all_patients', validate_seg_net_on_all_patients),
    ('ensemble_foldername_of_prob_maps', ['gen_prob_maps']), # False=gen_prob_maps else list of foldernames in datapipeline_directory
])

interpolate_candidates = OrderedDict([
    ('n_candidates', 20), #10
    ('new_spacing_zyx', [0.5, 0.5, 0.5]), # y, x, z
    ('new_data_type', 'uint8'),
    ('new_candidates_shape_zyx', [64, 64, 64]),
    ('crop_raw_scan_buffer', 10),
])

interpolate_candidates_res05 = OrderedDict([
    ('n_candidates', 10),
    ('new_spacing_zyx', [0.5, 0.5, 0.5]), # y, x, z
    ('new_data_type', 'uint8'),
    ('new_candidates_shape_zyx', [64, 64, 64]),
    ('crop_raw_scan_buffer', 10),
])

interpolate_candidates_res07 = OrderedDict([
    ('n_candidates', 10),
    ('new_spacing_zyx', [0.7, 0.7, 0.7]), # y, x, z
    ('new_data_type', 'uint8'),
    ('new_candidates_shape_zyx', [64, 64, 64]),
    ('crop_raw_scan_buffer', 10),
])

#------------------------------
#           80 20 submission
#------------------------------
gen_submission_2D_05res_80 = OrderedDict([
    ('splitting', 'submission'), # 'validation' or 'submission' or 'holdout'<-not implemented yet
    ('checkpoint_dir', './dsb3_networks/classification/resnet2D_0.5res_80/output_dir/old_but_gold_plane_mil0_b4_init_luna'),
    ('patients_lst_path', './datapipeline_final/dsb3_0/interpolate_candidates_res05/ho_patients.lst'), # if False->filter_candidates; if 'interpolate_candidates'->lst and arrays from current dataset is used
    ('num_augs_per_img', 0),
    ('sample_submission_lst_path', '/'.join(pipe['raw_data_dirs']['dsb3'].split('/')[:-2])  + '/stage2_sample_submission.csv'),
])
gen_submission_2D_07res_80 = OrderedDict([
    ('splitting', 'submission'), # 'validation' or 'submission' or 'holdout'<-not implemented yet
    ('checkpoint_dir', './dsb3_networks/classification/resnet2D_0.7res_80/output_dir/old_but_gold_plane_mil0_b4_init_luna'),
    ('patients_lst_path', './datapipeline_final/dsb3_0/interpolate_candidates_res07/ho_patients.lst'), # if False->filter_candidates; if 'interpolate_candidates'->lst and arrays from current dataset is used
    ('num_augs_per_img', 0),
    ('sample_submission_lst_path', '/'.join(pipe['raw_data_dirs']['dsb3'].split('/')[:-2])  + '/stage2_sample_submission.csv'),
])
gen_submission_3D_05res_80 = OrderedDict([
    ('splitting', 'submission'), # 'validation' or 'submission' or 'holdout'<-not implemented yet
    ('checkpoint_dir', './dsb3_networks/classification/resnet3D_0.5res_80/output_dir/3Dtest_c10_init'),
    ('patients_lst_path', './datapipeline_final/dsb3_0/interpolate_candidates_res05/ho_patients.lst'), # if False->filter_candidates; if 'interpolate_candidates'->lst and arrays from current dataset is used
    ('num_augs_per_img', 0),
    ('sample_submission_lst_path', '/'.join(pipe['raw_data_dirs']['dsb3'].split('/')[:-2])  + '/stage2_sample_submission.csv'),
])
gen_submission_3D_07res_80 = OrderedDict([
    ('splitting', 'submission'), # 'validation' or 'submission' or 'holdout'<-not implemented yet
    ('checkpoint_dir', './dsb3_networks/classification/resnet3D_0.7res_80/output_dir/3Dtest_c10_init'),
    ('patients_lst_path', './datapipeline_final/dsb3_0/interpolate_candidates_res07/ho_patients.lst'), # if False->filter_candidates; if 'interpolate_candidates'->lst and arrays from current dataset is used
    ('num_augs_per_img', 0),
    ('sample_submission_lst_path', '/'.join(pipe['raw_data_dirs']['dsb3'].split('/')[:-2])  + '/stage2_sample_submission.csv'),
])



#------------------------------
#           100 0 submission
#------------------------------
gen_submission_2D_05res_100 = OrderedDict([
    ('splitting', 'submission'), # 'validation' or 'submission' or 'holdout'<-not implemented yet
    ('checkpoint_dir', './dsb3_networks/classification/resnet2D_0.5res_100/output_dir/old_but_gold_plane_mil0_b4_init_luna'),
    ('patients_lst_path', './datapipeline_final/dsb3_0/interpolate_candidates_res05/ho_patients.lst'), # if False->filter_candidates; if 'interpolate_candidates'->lst and arrays from current dataset is used
    ('num_augs_per_img', 0),
    ('sample_submission_lst_path', '/'.join(pipe['raw_data_dirs']['dsb3'].split('/')[:-2])  + '/stage2_sample_submission.csv'),
])
gen_submission_2D_07res_100 = OrderedDict([
    ('splitting', 'submission'), # 'validation' or 'submission' or 'holdout'<-not implemented yet
    ('checkpoint_dir', './dsb3_networks/classification/resnet2D_0.7res_100/output_dir/old_but_gold_plane_mil0_b4_init_luna'),
    ('patients_lst_path', './datapipeline_final/dsb3_0/interpolate_candidates_res07/ho_patients.lst'), # if False->filter_candidates; if 'interpolate_candidates'->lst and arrays from current dataset is used
    ('num_augs_per_img', 0),
    ('sample_submission_lst_path', '/'.join(pipe['raw_data_dirs']['dsb3'].split('/')[:-2])  + '/stage2_sample_submission.csv'),
])
gen_submission_3D_05res_100 = OrderedDict([
    ('splitting', 'submission'), # 'validation' or 'submission' or 'holdout'<-not implemented yet
    ('checkpoint_dir', './dsb3_networks/classification/resnet3D_0.5res_100/output_dir/3Dtest_c10_init'),
    ('patients_lst_path', './datapipeline_final/dsb3_0/interpolate_candidates_res05/ho_patients.lst'), # if False->filter_candidates; if 'interpolate_candidates'->lst and arrays from current dataset is used
    ('num_augs_per_img', 0),
    ('sample_submission_lst_path', '/'.join(pipe['raw_data_dirs']['dsb3'].split('/')[:-2])  + '/stage2_sample_submission.csv'),
])
gen_submission_3D_07res_100 = OrderedDict([
    ('splitting', 'submission'), # 'validation' or 'submission' or 'holdout'<-not implemented yet
    ('checkpoint_dir', './dsb3_networks/classification/resnet3D_0.7res_100/output_dir/3Dtest_c10_init'),
    ('patients_lst_path', './datapipeline_final/dsb3_0/interpolate_candidates_res07/ho_patients.lst'), # if False->filter_candidates; if 'interpolate_candidates'->lst and arrays from current dataset is used
    ('num_augs_per_img', 0),
    ('sample_submission_lst_path', '/'.join(pipe['raw_data_dirs']['dsb3'].split('/')[:-2])  + '/stage2_sample_submission.csv'),
])






















# LUNA Stuff
filter_candidates = OrderedDict([
   ('n_candidates', 20),
   ('checkpoint_dir', '/media/niklas/Data_3/dsb3/dsb3a_networks/classification/luna_resnet2D/output_dir/gen8_20z_3rot_stage1_deep_prio3/'),
   ('num_augs_per_img', 1),
   ('all_patients', validate_seg_net_on_all_patients)
])


pred_cancer_per_candidate = OrderedDict([
    ('n_candidates', 20),
#    ('checkpoint_dir', assets_dir + 'checkpoints/pred_cancer_per_candidate/cross_3cands_720epochs/'),
    ('checkpoint_dir', '/media/niklas/Data_3/dsb3/dsb3a_networks/classification/luna_resnet2D/output_dir/gold_prio3_plane_mil0/' ),#'/media/niklas/Data_3/dsb3/dsb3a_networks/classification/resnet2D/output_dir/cv/c10_cv' + str(fold) + '/'
    ('num_augs_per_img', 1), # 1 means NO augmentation	
    ('all_patients', True),
    ('list_to_predict', '/media/niklas/Data_3/dsb3/datapipeline_gen9/dsb3_0/interpolate_candidates/cv5/cv/full.lst') #'/media/niklas/Data_3/dsb3/datapipeline_gen9/dsb3_0/interpolate_candidates/cv5/cv/va' + str(fold) + '.lst'
])

include_nodule_distr = OrderedDict([
   ('lists_to_predict',  ['/media/niklas/Data_3/dsb3/datapipeline_gen9/dsb3_0/pred_cancer_per_candidate/concat_features_tr.lst',
                        '/media/niklas/Data_3/dsb3/datapipeline_gen9/dsb3_0/pred_cancer_per_candidate/concat_features_va.lst',
                        ]),
   ('n_candidates', 20),
   ('bin_size', 0.05),
   ('kernel_width', 0.2),
   ('xg_max_depth', 2),
   ('xg_eta', 0.01),
   ('xg_num_round', 10000),
   ('sample_submission_lst_path', '../raw_data/dsb3/stage1_sample_submission.csv'),
])







# ------------------------------------------------------------------------------
# nodule segmentation parameters
# ------------------------------------------------------------------------------
# Adjusted Parameters for LUNA retraining 
# Checkpoints of those settings are already included in the April 7 Stage1 Model release
gen_nodule_masks = OrderedDict([
    ('ellipse_mode', True),
    ('reduced_mask_radius_fraction', 0.5),
    ('mask2pred_lower_radius_limit_px', 5),
    ('mask2pred_upper_radius_limit_px', 20),
    ('LUNA16_annotations_csv_path', './dsb3a_assets/LIDC-annotations_2_nodule-seg_annotations/annotations_min+missing_LUNA16_patients.csv'),
    ('yx_buffer_px', 0),
    ('z_buffer_px', 0),
])

gen_nodule_seg_data = OrderedDict([
    ('view_angles', [0]), # per view_plane (degree)
    ('extra_radius_buffer_px', 15),
    ('num_channels', 5),
    ('stride', 2),
    ('crop_size', [128, 128]),
    ('view_planes', 'yxz'), 
    ('num_negative_examples_per_nodule_free_patient_per_view_plane', 15),
    ('HU_tissue_range', [-1000, 400]), # MIN_BOUND, MAX_BOUND [-1000, 400]
])

# ------------------------------------------------------------------------------
# Eval parameters
# ------------------------------------------------------------------------------

gen_candidates_eval = OrderedDict([
    ('max_n_candidates', 20),
    ('max_dist_fraction', 0.5),
    ('priority_threshold', 3), 
    ('sort_candidates_by', 'prob_sum_min_nodule_size'), #prob_sum_min_nodule_size
    ('all_patients', True)
])

gen_candidates_vis = OrderedDict([
    ('inspect_what', 'true_positives')
])

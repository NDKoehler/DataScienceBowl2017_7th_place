# -----------------------------------------------------#
	     #Process LUNA for retraining
# -----------------------------------------------------#
python3.4 dsb3.py 0 -n 8 -s '' -ds 'LUNA16'
python3.4 dsb3.py 2 -n 8 -s '' -ds 'LUNA16'
python3.4 dsb3.py 6 -n 8 -s '' -ds 'LUNA16' #(bug) always uses entire data, will fail if previous steps use 10 patients

#Train Segmentation
cd ../dsb3_networks/nodule_segmentation/net/
python3.4 run.py stage1
python3.4 run.py stage2

# copy checkpoint to nodule segmentation
cp -r output_dir/128x128_5Channels_mutliview_stage2 ../../../checkpoints/nodule_segmentation
cd ../../../dsb3a/

python3.4 dsb3.py 1 -s '' -ds 'LUNA16'
python3.4 dsb3.py 3 -s '' -ds 'LUNA16'
python3.4 dsb3.py 4 -s '' -ds 'LUNA16'

#binaries luna candidate_lists
python3.4 binarize_candidates.py

#Train noduleness classifier
cd ../dsb3_networks/classification/luna_resnet2D/
python3.4 run.py 
cd ../luna_resnet3D/
python3.4 run.py 
cd ../../../dsb3a/

# -----------------------------------------------------#
	#Run dsb3 processing / model training
# -----------------------------------------------------#
python3.4 dsb3.py 0 -s ''
python3.4 dsb3.py 1 -s ''
python3.4 dsb3.py 3 -s ''

#make links
cd ../datapipeline_final/dsb3_0/
ln -s gen_candidates gen_candidates_res05
ln -s gen_candidates gen_candidates_res07
ln -s resample_lungs resample_lungs_res05
ln -s resample_lungs resample_lungs_res07
cd ../../dsb3a/

python3.4 dsb3.py 4 -s '_res05'
python3.4 dsb3.py 4 -s '_res07'

#reorder lists
python3.4 enforce_ordering.py

#100 - 0 split
cd ../dsb3_networks/classification/resnet2D_0.5res_100/
python3.4 run.py
cd ../resnet2D_0.7res_100/
python3.4 run.py
cd ../resnet3D_0.5res_100/
python3.4 run.py
cd ../resnet3D_0.7res_100/
python3.4 run.py

#80 - 20 split
cd ../resnet2D_0.5res_80/
python3.4 run.py
cd ../resnet3D_0.5res_80/
python3.4 run.py
cd ../resnet2D_0.7res_80/
python3.4 run.py
cd ../resnet3D_0.7res_80/
python3.4 run.py
cd ../../../dsb3a/

#sub 1
python3.4 dsb3.py 7 -s '_2D_05res_80'
python3.4 dsb3.py 7 -s '_2D_07res_80'
python3.4 dsb3.py 7 -s '_3D_05res_80'
python3.4 dsb3.py 7 -s '_3D_07res_80'

#sub 2
python3.4 dsb3.py 7 -s '_2D_05res_100'
python3.4 dsb3.py 7 -s '_2D_07res_100'
python3.4 dsb3.py 7 -s '_3D_05res_100'
python3.4 dsb3.py 7 -s '_3D_07res_100'

#combine submissions (average)
python3.4 combine_subs.py


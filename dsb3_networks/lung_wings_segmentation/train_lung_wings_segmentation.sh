python3 preprocessing_scripts/3_lst2tfrecord_segmentation.py --img_lst ./data/lsts+records/tr.lst --record_file ./data/lsts+records/tr.tfr --class_id 0 --resize 128x128
python3 preprocessing_scripts/3_lst2tfrecord_segmentation.py --img_lst ./data/lsts+records/va.lst --record_file ./data/lsts+records/va.tfr --class_id 0 --resize 128x128

cd ./net/
python3 run.py

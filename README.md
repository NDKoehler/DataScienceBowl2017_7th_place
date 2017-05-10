# Kaggle national datascience bowl 2017 7th place code

### Documentation
The model description can be found in ./documentation/DL_Munich_model_desc.pdf

### Operating system
* Ubuntu 14.04 

### The final submission are generated on the following system components
* GPU: Nvidia GTX 1080
* CPU: Intel(R) Core(TM) i7-4930K CPU
* RAM: 32GB of RAM
* Around 200GB of free Memory

### Package requirements
* opencv-python 3.2.0.6
* Python 3.4.3
* dicom 0.9.9-1
* joblib 0.10.3
* tensorflow-gpu 1.0.1
* SimpleITK 0.10.0.0
* numpy 1.12.0
* pandas 0.19.2
* scipy 0.18.1
* scikit-image 0.12.3
* scikit-learn 0.18.1


### Preparing the data
adjust raw_data_absolute_path in "params_niklas_fix.py" (line 6) to the raw dsb3 data directory. The raw dsb3 data directory is expected to contain the following folders and files:
* stage1/    (unzipped stage1.7z)
* stage2/    (unzipped stage2.7z)
* stage2_sample_submission.csv


adjust raw_LUNA_absolute_path in "params_niklas_fix.py" (line 7) to the raw LUNA data directory. The directory is expected to contain the following folders and files from the LUNA16 challenge (https://luna16.grand-challenge.org/data/):
* subset0.zip to subset9.zip: 10 zip files which contain all CT images
* annotations.csv: csv file that contains the annotations used as reference standard for the 'nodule detection' track
* sampleSubmission.csv: an example of a submission file in the correct format
* candidates_V2.csv: csv file that contains the candidate locations for the ‘false positive reduction’ track


The GPU ID and number of cores for multithreading can be adjusted in line 23,24 in "params_niklas_fix.py":
('n_CPUs', 4),
('GPU_ids', [0]),


Download the checkpoint folder from:
https://www.dropbox.com/sh/70dvei9ie7fpwpa/AADTU8pc8T5TzII38j5kstroa?dl=0
and extract it to the ./ directory


### Running entire pipeline
The intermediate steps will produce outputs in the ./datapipeline_final/ directory. The final 2 submissions will be placed in the ./out/ directory.

```
$ sh run_pipeline.sh
```




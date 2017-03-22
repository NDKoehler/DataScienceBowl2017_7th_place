## dsb3 pipeline

Navigate into the repository and call
```
$ ./dsb3.py -h
```
for getting help.

### Optimize the pipeline in subsequent runs

Let us start a first `run` of the pipeline and restrict ourselves to step 0 and 1 with
```
$ ./dsb3.py step0,step1
```
which produces the following output
```
$ cat ~/test/LUNA16_0/log.txt
2017-03-22 01:49 | 00:00:00 - run 0 / step0 (resample_lungs)
2017-03-22 01:49 | 00:00:11 - finished step
2017-03-22 01:49 | 00:00:00 - run 0 / step1 (gen_prob_maps)
2017-03-22 01:50 | 00:00:27 - finished step
```
Now we want to change a parameter for step1, but do not want to recompute step0, so we do
```
$ ./dsb3.py step1 --descr "now using 3 view angles"
```
which produces
```
$ cat ~/test/LUNA16_1/log.txt
2017-03-22 01:55 | 00:00:00 - run 1 / step1 (gen_prob_maps) with init 0
2017-03-22 01:55 | 00:00:16 - finished step
```
By providing a description, we automatically started a `run` 1 that retrieves
the data for `step0` from `run` 0 but keeps on producing new data in the higher
steps. Now `./dsb3.py -h` shows
````
...
Choices for "--run" and "--init_run":
  0   2017-03-22 01:49 : default
  1   2017-03-22 01:55 : now using 3 view_angles
```
This is the underlying directory structure.
```
$ ls ~/test/LUNA16_
LUNA16_0/         LUNA16_1/         LUNA16_runs.json 
$ ls ~/test/LUNA16_0/
gen_prob_maps/                log.txt                       patients_raw_data_paths.json  resample_lungs/               
$ ls ~/test/LUNA16_1/
gen_prob_maps/ log.txt        
```
On the step-level, the directory structure is the same for each step. The
step-level logs show more fine grained information than the run-level logs
above, which track global progress and errors.
```
$ cat ~/test/LUNA16_1/gen_prob_maps/
arrays/      figs/        log_tf.txt   log.txt      out.json     params.json  
$ cat ~/test/LUNA16_1/gen_prob_maps/log.txt 
00:00:00 - started with default init: most recent run
00:00:16 - finished step
$ cat ~/test/LUNA16_0/resample_lungs/log.txt 
00:00:00 - default init: most recent run"
00:00:00 - resizing and interpolating scans
00:00:03 - segmenting lung wings and cropping the scan
00:00:07 - finished step
```
Invoking help shows you more possibilities: you can combine different runs using `--run_init i` and
rerun old runs using `--run i` while optionally update their description using `--descr d`.

### Visualization and Evaluation
This will be based on markdown, html and runnable as `tensorboard` on a local
server. I know what to do, but it cost me lot's of time to solve all kinds of
small things (like stopping tensorflow from outputting loads and loads of
useless variables. Comes tomorrow and will work in this direction visualize the
result of a computation using
```
./dsb3.py step0 -a vis
```
Run and visualize a sequence of steps
```
./dsb3.py step0,step1,step2 -a all
```

### 

### Features
* manage pipeline optimization via `pipeline runs`
* easy to use mit top-level command, z.b. `dsb3 step0` oder `dsb3 resample_lungs` 
  fuehrt “step0" aus, `dsb3 step0 —action evaluate` evaluiert den schritt usw.
* trennung pipeline und steps, pipeline modul regelt alles uebergeordnete, step
  module sind untergeordnet und muessen bestimmte constraints erfuellen, dadurch
  werden unter anderem parameter automatisch gecheckt, directories und output
  files gemanagt, fehler einheitlich getrackt, usw.
* einheitliches logging, getrennte `pipeline` log und `step` logs, mit info,
  warning und error levels, alles laeuft in jedem fall durch auch wenn ein patient
  vollstaendig korrumpiert ist
* ein master config file fuer jeden user
* ein modul `tf_tools` das checkpoints laedt, modelle initialisiert usw.
* ein modul `tf_models` das tf modelle speichert
* ein ordner `checkpoints` im parent vom dsb3 package, der die step struktur
  erfuellt und aus den `tf_models` geschrieben wird, diese regelung wird noch
  verbessert
* human-readable jsons enable simple checking of json files: [dsb3/hrjson](dsb3/hrjson)

### Contribution Guidelines
* code nach https://www.python.org/dev/peps/pep-0008/
* docstrings im numpy/scipy style http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
* __kein__ generelles 80-Zeilen limit, weil ihr ja dicke Screens habt, wenn ihr
  euch davon ueberzeugen lasst, wuerds mich allerdings auch freuen, oder zumindest 120

### Minor updates 
* meaningful pipeline environment settings
```python
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in master_config['GPU_ids']])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tensorflow info and warning logs
```
* replaced deprecated `tf.initialize_all_variables()` with `tf.global_variables_initializer()`

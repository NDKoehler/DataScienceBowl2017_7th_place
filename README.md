## dsb3 pipeline

Currently, the pipeline is operated via the script `dsb3.py`. Navigate into the
repository using `cd` and call
```
./dsb3.py --help
```
for getting help, or
```
./dsb3.py step0
```
for running step 0.

### Features
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

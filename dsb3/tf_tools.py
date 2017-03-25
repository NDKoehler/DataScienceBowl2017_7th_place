import os, sys
import logging
import json
from contextlib import contextmanager
from . import pipeline as pipe

def load_network(checkpoint_dir, image_shape=None, reuse=None, batch_size=None):
    with redirect_stdout():
        config = json.load(open(checkpoint_dir + '/config.json'))
        if image_shape is None:
            image_shape = config['image_shape']
        # global imports
        import tensorflow as tf
        from tensorflow.python.training import saver as tf_saver
        # in case we have run a tensorflow computation before, we need to reset the
        # default graph to clear all variables
        if reuse is None:
            tf.reset_default_graph()
        # checkpoint imports
        sys.path.append(checkpoint_dir + '/architecture/')
        model = __import__(config['model_name'] + '_model')
        model = getattr(model, config['model_name'])
        # from basic_logging import initialize_logger
        # out_dir = pipe.step_dir + 'tf/'
        # initialize_logger(folder=out_dir)
        # if not tf.gfile.Exists(out_dir): # create a new eval directory, where to save predictions, all.log and config.json
        #     tf.gfile.MakeDirs(out_dir)
        config['VARIABLES_TO_RESTORE'] = tf.contrib.slim.get_variables_to_restore()
        config['UPDATE_OPS_COLLECTION'] = tf.GraphKeys.UPDATE_OPS
        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=pipe.GPU_memory_fraction),
                                                allow_soft_placement=True,
                                                log_device_placement=False))
        sess.run(init_op)
        data = {'images': tf.placeholder(dtype=tf.float32, shape=[batch_size] + image_shape, name='image_placeholder')}
        with tf.device('/gpu:' + os.environ['CUDA_VISIBLE_DEVICES']):
            config['dropout'] = 1.0
            output, endpoints = model(data=data,
                                      reuse=reuse,
                                      restore_logits=False,
                                      is_training=False,
                                      scope=config['model_name'],
                                      **config)
         # add all endpoints to predict
        config['endpoints'] = ['probs', 'logits']
        pred_ops = {}
        for key in config['endpoints']:
            if key in endpoints and key not in pred_ops:
                pred_ops[key] = endpoints[key]
            elif key in output and key not in pred_ops:
                pred_ops[key] = output[key]
        if len(pred_ops) != len(config['endpoints']):
            raise ValueError('Not all enpoints found in the graph!: {}'.format(config['endpoints']))
        # restore checkpoint and create saver
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        var_lst = []
        for var in tf.global_variables():
            if var.name.split('/')[0] == config['model_name']:
                var_lst.append(var)
        saver = tf_saver.Saver(var_lst, write_version=tf.train.SaverDef.V2)
        # the following outputs all variables in the graph from _pywrap_tensorflow.so
        # when calling _pywrap_tensorflow.TF_ExtendGraph
        # this is why we use redirect_std_out() context manager
        saver.restore(sess, ckpt.model_checkpoint_path)
    return sess, pred_ops, data

@contextmanager
def redirect_stdout(to=pipe.log_tf):
    """from http://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirect stdout
        finally:
            # restore stdout. buffering and flags such as CLOEXEC may be different
            _redirect_stdout(to=old_stdout)

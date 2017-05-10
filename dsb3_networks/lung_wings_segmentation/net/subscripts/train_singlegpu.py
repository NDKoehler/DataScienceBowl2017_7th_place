import sys
import os
import math
import time
from datetime import datetime
import numpy as np

import tensorflow as tf
from hellsicht.tensorflow.modules import metrics
import logging

from tensorflow.python.ops import variables
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.platform import resource_loader


def restore_checkpoint(session, model_path, var_list=tf.all_variables(), ignore_missing_vars=True, reshape_variables=False):
    from tensorflow.python import pywrap_tensorflow
    if ignore_missing_vars:
        reader = pywrap_tensorflow.NewCheckpointReader(model_path)
        if isinstance(var_list, dict):
            var_dict = var_list
        else:
            var_dict = {var.op.name: var for var in var_list}
        available_vars = {}
        for var in var_dict:
            if reader.has_tensor(var):
                available_vars[var] = var_dict[var]
            else:
                logging.warning(
                    'Variable %s missing in checkpoint %s', var, model_path)
        var_list = available_vars
    saver = tf_saver.Saver(
        var_list, reshape=reshape_variables, write_version=tf.train.SaverDef.V2)

    def callback(session):
        saver.restore(session, model_path)
    return callback, saver


def should_stop(coord):
    if coord != None:
        return coord.should_stop()
    else:
        return False


def create_summary_op(name):

    placeholder = tf.placeholder(dtype=tf.float32)
    summary_op = tf.summary.scalar(name, placeholder)

    return (summary_op, placeholder)


def train(*args, **kwargs):
        # Get all neccessary paramters from kwargs
    try:
            # Get model graph
        my_model_graph = kwargs['model_graph']
    except:
        logging.error('(model_graph) was not provided!')
        raise KeyError('(model_graph) was not provided!')
    try:
        # Get loss operations
        my_loss = kwargs['loss']
    except:
        logging.error('(losses) was not provided!')
        raise KeyError('(losses) was not provided!')

    try:
        # Get metric operations
        my_metric_ops = kwargs['metrics']
    except:
        my_metric_ops = None
        pass

    # Build the summary operation based on the TF collection of Summaries.
    if not kwargs['output_dir'] or 'output_dir' not in kwargs:
        kwargs['output_dir'] = 'output_dir/train_dir/%s' % datetime.now().strftime(
            '%Y_%m_%d_%H.%M')
    logging.info(
        'Saving evaluation results to: {}'.format(kwargs['output_dir']))

    # Add train iterator
    train_iter = kwargs['train_iter']
    train_iter.initialize()
    train_data = train_iter.data_batch()
    train_label = train_iter.label_batch()
    # Add validation iterator
    valid_iter = kwargs['valid_iter']
    valid_iter.initialize()
    valid_data = valid_iter.data_batch()
    valid_label = valid_iter.label_batch()

    # Define global step
    global_step = tf.get_variable(
        'global_step', [],  initializer=tf.constant_initializer(0), trainable=False)

    test_image_to_log = tf.placeholder(
        tf.uint8, [40,kwargs['image_shape'][-3], kwargs['image_shape'][-2], 3])
    log_image_test = tf.summary.image("Test examples", test_image_to_log, max_outputs = 40)

    train_image_to_log = tf.placeholder(
        tf.uint8, [40,kwargs['image_shape'][-3], kwargs['image_shape'][-2], 3])
    log_image_train = tf.summary.image("Train examples", train_image_to_log, max_outputs = 40)


    # Selecte optimizer
    lr = kwargs['learning_rate']
    if kwargs['optimizer'] == 'GradientDescentOptimizer':
        opt = tf.train.GradientDescentOptimizer(kwargs['learning_rate'])
    elif kwargs['optimizer'] == 'MomentumOptimizer':
        opt = tf.train.MomentumOptimizer(
            kwargs['learning_rate'], kwargs['momentum'])
    elif kwargs['optimizer'] == 'AdamOptimizer':
        opt = tf.train.AdamOptimizer(kwargs['learning_rate'])
    elif kwargs['optimizer'] == 'AdadeltaOptimizer':
        opt = tf.train.AdadeltaOptimizer(
            kwargs['learning_rate'], kwargs['rho'])
    elif kwargs['optimizer'] == 'RMSPropOptimizer':
        decay_steps = int(kwargs['tr_num_examples'] / kwargs[
                          'batch_size'] * kwargs['num_epochs_per_decay'])
        lr = tf.train.exponential_decay(kwargs['learning_rate'],
                                        global_step,
                                        decay_steps,
                                        kwargs['learning_rate_decay_factor'],
                                        staircase=True)
        opt = tf.train.RMSPropOptimizer(lr, kwargs['RMSPROP_DECAY'],
                                        momentum=kwargs['momentum'],
                                        epsilon=kwargs['RMSPROP_EPSILON'])
    else:
        logging.error('Hyperparameter "optimizer" was not provided!')
        raise KeyError('Hyperparameter "optimizer" was not provided!')
    logging.info('Selected Optimizer: {}'.format(kwargs['optimizer']))

    gpu_id = kwargs['gpus']
    if not isinstance(gpu_id, list):
        gpu_id = [gpu_id]
    gpu_id = gpu_id[0]

    with tf.device('/gpu:%d' % gpu_id):
        logging.info('Training on gpu:{}'.format(gpu_id))
        # Get endpoint / or get tensor from session.graph
        out_, train_eps_ = my_model_graph(train_data,
                                          restore_logits=False,
                                          is_training=True,
                                          reuse=None,
                                          **kwargs)

        # Add loss operation
        loss_op = my_loss(out_, train_label, 'train', **kwargs)
        # train_metric_ops = tf.group(*[m(out_, train_label, 'train', **kwargs)
        #        for m in my_metric_ops])

        # Add loss-averages for training
        tr_loss_averages_op = metrics.add_loss_averages(
            tf.get_collection('train'), 'train_summaries')

        # Add learning rate to summary
        train_summaries = [tf.summary.scalar('learning_rate', lr)]
        train_summaries += tf.get_collection('train_summaries')

        # Calculate and apply selected gradients
        if kwargs['train_scopes']:
            ws = []
            # Find all parameters in the train scopes
            for tr_scope in kwargs['train_scopes']:
                logging.info('Add to training endpoints: {}'.format(tr_scope))
                with tf.variable_scope(tr_scope, reuse=True) as scope:
                    w_names = ['/'.join(i.name.split('/')[1:])[:-2]
                               for i in tf.get_collection(
                               key=tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope=scope.name)]
                    ws += [tf.get_variable(w_name) for w_name in w_names]
                    for w_name in w_names:
                        logging.info(
                            '({})-paramter: {}'.format(tr_scope, w_name))

            # Compute gradients for this selected parameters
            grads = opt.compute_gradients(loss_op, ws)
            apply_gradient_op = opt.apply_gradients(
                grads, global_step=global_step)
        else:
            # Update all parameters
            logging.info('Adding all parameters to training endpoints')
            grads = opt.compute_gradients(loss_op)
            apply_gradient_op = opt.apply_gradients(
                grads, global_step=global_step)

        # Get batchnorm moving mean and variance updates
        if 'UPDATE_OPS_COLLECTION' in kwargs:
            logging.debug('add batchnorm updates')
            batchnorm_updates = tf.get_collection(
                kwargs['UPDATE_OPS_COLLECTION'])
            batchnorm_updates_op = tf.group(*batchnorm_updates)

        # Add histograms for gradients.
        #for grad, var in grads:
        #    if grad is not None:
        #        train_summaries.append(
        #            tf.histogram_summary(var.op.name + '/gradients', grad))

        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        variable_averages = tf.train.ExponentialMovingAverage(
            kwargs['MOVING_AVERAGE_DECAY'], global_step)

        # Update moving averages of all parameters
        variables_to_average = (
            tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)
        # Group all updates
        if 'UPDATE_OPS_COLLECTION' in kwargs:
            logging.debug('batchnorm updates in train_op')
            train_op = tf.group(
                apply_gradient_op, variables_averages_op, batchnorm_updates_op)
        else:
            logging.debug('no batchnorm updates in train_op')
            train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Add evaluation graph after training step
        test_out_, _ = my_model_graph(valid_data,
                                      restore_logits=False,
                                      is_training=False,
                                      reuse=True,
                                      **kwargs)

                

        # Add validation metrics and averages
        test_loss_op = my_loss(test_out_, valid_label, 'validation', **kwargs)
        # test_metric_ops = tf.group(*([m(test_out_, valid_label, 'validation', **kwargs)
        #        for m in my_metric_ops]))
        if my_metric_ops != None:
            test_metric_ops_list = my_metric_ops(
                test_out_, valid_label, 'validation', **kwargs)

        # Add loss-averages for validation
        va_loss_averages_op = metrics.add_loss_averages(
            tf.get_collection('validation'),
            'validation_summaries')

        validation_summaries = tf.get_collection('validation_summaries')

    # Build summary operation
    train_summary_op = tf.summary.merge(train_summaries)
    validation_summary_op = tf.summary.merge(validation_summaries)
    # summary_op = tf.merge_all_summaries()
    # Build an initialization operation to run below.
    init_op = tf.initialize_all_variables()

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=kwargs['gpu_fraction'])

    # Define a session
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=kwargs['log_device_placement'])
    )

    # initialize all variables
    sess.run(init_op)

    # restore checkpoint and create saver
    if kwargs['pretrained_checkpoint_dir']:
        ckpt = tf.train.get_checkpoint_state(
            kwargs['pretrained_checkpoint_dir'])
        ignore_missing_vars = True
        print ('----------------\nrestoring checkpoint: {} ignore_missing_vars={}'.format(
            ckpt.model_checkpoint_path, ignore_missing_vars))
        init_fn, _ = restore_checkpoint(
            sess, ckpt.model_checkpoint_path, var_list=tf.all_variables(), ignore_missing_vars=ignore_missing_vars, reshape_variables=False)
        init_fn(sess)
        print ('checkpoint restored: {} ignoring missing vars={}\n------------------------'.format(
            ckpt.model_checkpoint_path, ignore_missing_vars))
    # else:

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    # Start the queue runners.
    coord = None
    if train_iter.need_queue_runners() or valid_iter.need_queue_runners():
        logging.debug('Create coordinator, start queue runners...')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(
        kwargs['output_dir'],
        graph=sess.graph)

    # validation and testloss placeholder, manual rate, not every batch not
    # every epoch
    tr_summary_op_train, tr_loss_placeholder = create_summary_op(
        loss_op.name.split(':')[0])  # loss_op.name
    if my_metric_ops == None:
        va_summary_op_train, va_loss_placeholder = create_summary_op(
            test_loss_op.name.split(':')[0])  # test_loss_op.name
    else:
        va_summary_op_train_list, va_loss_placeholder_list = [], []
        for cnt, m in enumerate(test_metric_ops_list):
            if cnt == 10:
                va_summary_op_train, va_loss_placeholder = create_summary_op(
                    'va/' + test_loss_op.name.split(':')[0] + '_average_' + str(cnt))  # test_loss_op.name
            else:
                va_summary_op_train, va_loss_placeholder = create_summary_op(
                    'va/' + test_loss_op.name.split(':')[0] + '_channelid_' + str(cnt))  # test_loss_op.name

            va_summary_op_train_list.append(va_summary_op_train)
            va_loss_placeholder_list.append(va_loss_placeholder)

    for step in range(kwargs['max_steps']):

        epoch_start = time.time()
        # Training step
        if step % 1 == 0:
            num_iter = int(
                math.ceil(float(kwargs['tr_num_examples']) / kwargs['batch_size']))
            train_step = 0
            train_loss = []
            start_time = time.time()
            while train_step < num_iter and not should_stop(coord):
                train_iter.read_batch()
                train_data_feed = train_iter.get_data_batch()
                train_label_feed = train_iter.get_label_batch()
                if train_data_feed != None and train_label_feed != None:
                    # Merge data and label dicts
                    train_data.update(train_label)
                    train_data_feed.update(train_label_feed)
                    data_keys = train_data.keys()
                    data_feed_keys = train_data_feed.keys()
                    assert data_keys == data_feed_keys
                    train_loss_, _ = sess.run([loss_op, train_op],
                                              feed_dict={train_data[k]: train_data_feed[k] for k in data_keys})

                    
                    '''
                    print(train_data_feed['labels'].shape)
                    print(tmp_out['predictions'])
                    print(tmp_out['predictions'].shape)
                    print(train_loss_)
                    print(train_loss_.shape)
                    sys.exit()
                    '''
                else:
                    train_loss_, _ = sess.run([loss_op, train_op])
                assert not np.isnan(
                    train_loss_), 'Model diverged with training-loss = NaN'
                train_loss += [train_loss_]
                train_step += 1

            mean_loss_tr = np.mean(train_loss)

            summary_str = sess.run(
                tr_summary_op_train, feed_dict={tr_loss_placeholder: mean_loss_tr})
            summary_writer.add_summary(
                summary_str, (step * kwargs['tr_num_examples']))

            duration = time.time() - start_time
            examples_per_sec = kwargs['batch_size'] / float(duration)
            format_str = ('Epoch %d, tr_loss = %.5f (%.1f examples/sec; %.3f '
                          'sec/epoch)')
            logging.info(
                format_str % (step, mean_loss_tr, examples_per_sec, duration))

        # Evaluation step
        evaluation_step = 1 if 'evaluation_step' not in kwargs else kwargs[
            'evaluation_step']
        if step % evaluation_step == 0:
            # sess.run(batchnorm_updates_op)
            num_iter = int(
                math.ceil(float(kwargs['va_num_examples']) / kwargs['batch_size']))
            test_step = 0
            test_loss = []
            start_time = time.time()
            while test_step < num_iter and not should_stop(coord):
                valid_iter.read_batch()
                valid_data_feed = valid_iter.get_data_batch()
                valid_label_feed = valid_iter.get_label_batch()
                if valid_data_feed != None and valid_label_feed != None:
                    # Merge data and label dicts
                    valid_data.update(valid_label)
                    valid_data_feed.update(valid_label_feed)
                    data_keys = valid_data.keys()
                    data_feed_keys = valid_data_feed.keys()
                    assert data_keys == data_feed_keys
                    if my_metric_ops == None:
                        test_loss_ = sess.run([test_loss_op],
                                              feed_dict={valid_data[k]: valid_data_feed[k] for k in data_keys})
                        assert not np.isnan(
                            test_loss_), 'Model diverged with validation-loss = NaN'
                    else:
                        test_loss_ = sess.run(test_metric_ops_list,
                                              feed_dict={valid_data[k]: valid_data_feed[k] for k in data_keys})
                else:
                    if my_metric_ops == None:
                        test_loss_ = sess.run([test_loss_op])
                        assert not np.isnan(
                            test_loss_), 'Model diverged with validation-loss = NaN'
                    else:
                        test_loss_ = sess.run(test_metric_ops_list)

                test_loss += [test_loss_]
                test_step += 1

            if my_metric_ops == None:
                mean_loss_va = np.mean(test_loss)

                summary_str = sess.run(
                    va_summary_op_train, feed_dict={va_loss_placeholder: mean_loss_va})
                summary_writer.add_summary(
                    summary_str, (step * kwargs['tr_num_examples']))

                duration = time.time() - start_time
                examples_per_sec = kwargs['batch_size'] / float(duration)
                format_test_str = ('Epoch %d, va_loss = %.5f (%.1f examples/sec, %.3f '
                                   'sec/epoch)')
                logging.info(
                    format_test_str % (step, mean_loss_va, examples_per_sec, duration))
            else:
                test_loss = np.array(test_loss)
                test_loss = np.mean(test_loss, axis=0)

                for cnt, l in enumerate(test_loss):
                    summary_str = sess.run(va_summary_op_train_list[
                                           cnt], feed_dict={va_loss_placeholder_list[cnt]: l})
                    summary_writer.add_summary(
                        summary_str, (step * kwargs['tr_num_examples']))

                duration = time.time() - start_time
                examples_per_sec = kwargs['batch_size'] / float(duration)
                format_test_str = ('Epoch %d, va_loss_total = %.5f (%.1f examples/sec, %.3f '
                                   'sec/epoch)')
                logging.info(
                    format_test_str % (step, test_loss[0], examples_per_sec, duration))

        # IMAGE SUNMMARY STUFF
        summary_step = 1 if 'summary_step' not in kwargs else kwargs[
            'summary_step']
        if step % summary_step == 0:
            logging.debug('Add summary string...')
            if train_data_feed != None and train_label_feed != None:
                # Run all output-opterations and summary ops

                out_.update({'train_summary_op': train_summary_op})
                out = sess.run(
                    out_, feed_dict={train_data[k]: train_data_feed[k] for k in data_keys})
                # summary_str = out['train_summary_op']

                # Add image summaries
                if 'train_image_summary' in kwargs:
                    out.update(train_data_feed)
                    out.update({'step': step, 'mode': 'train'})
                    img_logs = kwargs['train_image_summary'](kwargs, **out)
                    
                    list_of_log_images = []
                    for train_output_to_log, name in img_logs:
                        list_of_log_images.append(train_output_to_log)
                    
                    feed = {
                        train_image_to_log: np.array(list_of_log_images)}
                    train_image_summary_str = sess.run(
                        log_image_train, feed_dict=feed)
                    summary_writer.add_summary(train_image_summary_str)

            else:
                # print("should not happen")
                # sys.exit()
                # Run all output operations and summary ops
                out_.update({'train_summary_op': train_summary_op})
                # Add input data and labels to this run
                out_.update(train_data)
                out_.update(train_label)
                out = sess.run(out_)

                if 'train_image_summary' in kwargs:
                    # Add current step and mode to image summary fuction input
                    out.update({'step': step, 'mode': 'train'})
                    img_logs = kwargs['train_image_summary'](kwargs, **out)
                    
                    list_of_log_images = []
                    for train_output_to_log, name in img_logs:
                        list_of_log_images.append(train_output_to_log)
                    
                    feed = {
                        train_image_to_log: np.array(list_of_log_images)}
                    train_image_summary_str = sess.run(
                        log_image_train, feed_dict=feed)
                    summary_writer.add_summary(train_image_summary_str)
                    
                    '''
                    for train_output_to_log, name in img_logs:
                        feed = {
                            test_image_to_log: train_output_to_log, log_image_name: name}
                        train_image_summary_str = sess.run(
                            log_image, feed_dict=feed)
                        summary_writer.add_summary(train_image_summary_str)
                    '''
            if valid_data_feed != None and valid_label_feed != None:

                test_out_.update(
                    {'validation_summary_op': validation_summary_op})
                out = sess.run(test_out_, feed_dict={
                               valid_data[k]: valid_data_feed[k] for k in data_keys})

                # Add image summaries
                if 'validation_image_summary' in kwargs:
                    out.update(valid_data_feed)
                    out.update({'step': step, 'mode': 'validation'})
                    img_logs = kwargs[
                        'validation_image_summary'](kwargs, **out)

                    list_of_log_images = []
                    for test_output_to_log, name in img_logs:
                        list_of_log_images.append(test_output_to_log)
                    
                    feed = {
                        test_image_to_log: np.array(list_of_log_images)}
                    test_image_summary_str = sess.run(
                        log_image_test, feed_dict=feed)
                    summary_writer.add_summary(test_image_summary_str)


            else:
                # print("should not happen")
                # sys.exit()
                # Run all output operations and the summary ops
                test_out_.update(
                    {'validation_summary_op': validation_summary_op})
                # Add input data and labels to summary run
                test_out_.update(valid_data)
                test_out_.update(valid_label)
                out = sess.run(test_out_)

                if 'validation_image_summary' in kwargs:
                    out.update({'step': step, 'mode': 'validation'})
                    img_logs = kwargs[
                        'validation_image_summary'](kwargs, **out)
                    
                    
                    list_of_log_images = []
                    for test_output_to_log, name in img_logs:
                        list_of_log_images.append(test_output_to_log)
                    
                    feed = {
                        test_image_to_log: np.array(list_of_log_images)}
                    test_image_summary_str = sess.run(
                        log_image_test, feed_dict=feed)
                    summary_writer.add_summary(test_image_summary_str)
                    
 
        # Save the model checkpoint periodically.
        save_step = 1 if 'save_step' not in kwargs else kwargs['save_step']
        if (step % save_step == 0 or (step + 1) == kwargs['max_steps']) and step != 0:
            logging.info(
                'Saving checkpoint to: {}, step: {}'.format(kwargs['output_dir'], step))
            checkpoint_path = os.path.join(
                kwargs['output_dir'], 'new-model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

        logging.info('Time per Epoch: {}'.format(time.time() - epoch_start))

    # Join threads and close session
    if train_iter.need_queue_runners() or valid_iter.need_queue_runners():
        logging.debug('request coordinater stop, joining threads...')
        coord.request_stop()
        coord.join(threads)
        sess.close()

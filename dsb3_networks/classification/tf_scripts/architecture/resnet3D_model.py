import re
import sys
import numpy as np

import tensorflow as tf
slim = tf.contrib.slim

from model_def.resnet3D import resnet3D_graph

TOWER_NAME = 'tower'


def load_resnet3D(
    kwargs, images, num_classes, dropout_keep_prob, kernel_num, is_training=False, restore_logits=False,
        scope=None, reuse=None, batch_norm_var_collection='moving_vars'):

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': kwargs['MOVING_AVERAGE_DECAY'],
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # scale
        'scale': kwargs['BATCH_NORM_SCALE'],
        # center
        'center': kwargs['BATCH_NORM_CENTER'],
        # collection containing update_ops.
        'updates_collections': None,                # is controlled in train_singlegpu by update_batch_norm_op
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }


    stddev = 0.1
    weight_decay = 0.00000004

    if kwargs['weights_initializer'] == 'truncated_normal_initializer':
        weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
        print ('weights_initializer: tf.truncated_normal_initializer')

    elif kwargs['weights_initializer'] == 'xavier_initializer':
        weights_initializer = tf.contrib.layers.xavier_initializer()
        print ('weights_initializer: tf.contrib.layers.xavier_initializer()')

    elif kwargs['weights_initializer'] == 'xavier_initializer_conv2d':
        weights_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        print (
            'weights_initializer: tf.contrib.layers.xavier_initializer_conv2d()')
    else:
        print ('ERROR: no weights_initializer has been chosen')
        sys.exit()

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected]):

        with slim.arg_scope([slim.conv2d],
                            reuse=reuse,
                            trainable=is_training,
                            weights_initializer=weights_initializer,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            activation_fn=tf.nn.relu):

            with slim.arg_scope([slim.conv2d_transpose],
                                reuse=reuse,
                                trainable=is_training,
                                weights_initializer=weights_initializer,
                                weights_regularizer=slim.l2_regularizer(
                                    weight_decay),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                activation_fn=tf.nn.relu):



                    logits, endpoints = resnet3D_graph(kwargs=kwargs, inputs=images,
                                                dropout_keep_prob=dropout_keep_prob,
                                                kernel_num=kernel_num,
                                                num_classes=num_classes,
                                                is_training=is_training,
                                                restore_logits=restore_logits,
                                                scope=scope,
                                                reuse=reuse)

    # Add summaries for viewing model statistics on TensorBoard.
    # summary_name = 'train_summaries' if is_training else 'validation_summaries'
    #_activation_summaries(endpoints, summary_name)
    # Grab the logits associated with the side head. Employed during training.
    # auxiliary_logits = endpoints['aux_logits']
    return logits, endpoints  # auxiliary_logits


def resnet3D(data, num_classes=2, restore_logits=False, is_training=True, scope='unet', reuse=None, **kwargs):
    # Load inception graph
    images = data['images']
    logits, endpoint = load_resnet3D(
        kwargs, images=images, num_classes=num_classes, dropout_keep_prob=kwargs[
            'dropout_keep_prob'], kernel_num=kwargs['kernel_num'],
        restore_logits=restore_logits, is_training=is_training, scope=scope, reuse=reuse)
    # Get endpoint / or get tensor from session.graph
    return {'logits': logits, 'probs': endpoint['probs']}, endpoint

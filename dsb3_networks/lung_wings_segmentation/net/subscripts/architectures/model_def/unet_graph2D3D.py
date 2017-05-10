    # Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import tensorflow as tf
slim = tf.contrib.slim

def print_shape(tensor):
    shape = tensor.get_shape()
    display_shape = [int(shape[x]) for x in range(len(shape))]
    print(tensor.name, display_shape)

def unet_graph2D3D(kwargs,
               inputs,
               dropout_keep_prob=0.8,
               kernel_num=32,
               num_classes=2,
               is_training=True,
               restore_logits=True,
               scope='unet2D3D',
               reuse=None):
    """
    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    dropout_keep_prob: dropout keep_prob.
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: Optional scope for op_scope.

    Returns:
    a list containing 'logits', 'aux_logits' Tensors.
    """

    # check unet_type on valid entry:
    list_of_valid_unet_types = ['standard', 'double_conv']
    if kwargs['unet_type'] not in list_of_valid_unet_types:
        print ('ERROR: unet_type not valid:', kwargs[
               'unet_type'], 'valid unet_types are:', list_of_valid_unet_types)
        sys.exit()

    ####################################################
    shape = inputs.get_shape()

    inputs3D = tf.expand_dims(tf.transpose(inputs, perm=[0,3,1,2]),4) # img_shape = [batch_size, y, x, z] -> [batch_size, z, y, x, 1]

    #feature stage
    with tf.variable_scope('3D-stage', reuse=reuse):
        net = slim.conv2d(inputs3D,      16 , [3, 3, 3], stride=1, scope='conv1',   reuse=reuse, data_format = 'NDHWC')
        net = slim.conv2d(net,         32 , [3, 3, 3], stride=1, scope='conv2',   reuse=reuse, data_format = 'NDHWC')
        net = slim.conv2d(net,         64 , [3, 3, 3], stride=1, scope='conv3',   reuse=reuse, data_format = 'NDHWC')
        print_shape(net)

    print_shape(inputs3D)
    sys.exit()
    inputs2D = tf.squeeze(net,2 )


    # [batch_size, z, y, x, 1] -> [batch_size, y, x, z]

    ####################################################

    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}
    with tf.variable_scope(scope, [inputs], reuse=reuse):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d], stride=1, padding='SAME'):
                    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.relu):

                        with tf.variable_scope('in_240x240x32', reuse=reuse):
                            in240 = slim.conv2d(inputs,         kernel_num, [
                                3, 3], stride=1, scope='conv_240x240x32_0',   reuse=reuse)
                            if kwargs['unet_type'] == 'double_conv':
                                in240 = slim.conv2d(
                                    in240,          kernel_num, [3, 3],
                                    stride=1, scope='conv_240x240x32_1',   reuse=reuse)
                        with tf.variable_scope('in_120x120x64', reuse=reuse):
                            in120 = slim.max_pool2d(
                                in240,            [2, 2], stride=2, scope='pool_120x120')
                            in120 = slim.conv2d(in120,          kernel_num * 2, [
                                3, 3], stride=1, scope='conv_120x120x64_0',   reuse=reuse)

                            if kwargs['unet_type'] == 'double_conv':
                                in120 = slim.conv2d(
                                    in120,          kernel_num * 2,
                                    [3, 3], stride=1, scope='conv_120x120x64_1', reuse=reuse)
                        with tf.variable_scope('in_60x60x128', reuse=reuse):
                            in60 = slim.max_pool2d(
                                in120,            [2, 2], stride=2, scope='pool_60x60')
                            in60 = slim.conv2d(in60,          kernel_num * 4, [
                                3, 3], stride=1, scope='conv_60x60x128_0',     reuse=reuse)

                            if kwargs['unet_type'] == 'double_conv':
                                in60 = slim.conv2d(
                                    in60,          kernel_num * 4,
                                    [3, 3], stride=1, scope='conv_60x60x128_1', reuse=reuse)
                        with tf.variable_scope('in_30x30x256', reuse=reuse):
                            in30 = slim.max_pool2d(
                                in60,             [2, 2], stride=2, scope='pool_30x30')
                            in30 = slim.conv2d(in30,          kernel_num * 8, [
                                3, 3], stride=1, scope='conv_30x30x256_0',     reuse=reuse)
                            if kwargs['unet_type'] == 'double_conv':
                                in30 = slim.conv2d(
                                    in30,          kernel_num * 8,
                                    [3, 3], stride=1, scope='conv_30x30x256_1',
                                    reuse=reuse)
                        with tf.variable_scope('in_and_out_15x15x512', reuse=reuse):
                            in_and_out15 = slim.max_pool2d(
                                in30,                [2, 2], stride=2, scope='pool_15x15')

                            in_and_out15 = slim.conv2d(in_and_out15,     kernel_num * 16, [
                                3, 3], stride=1, scope='conv_15x15x512_0',     reuse=reuse)
                            in_and_out15 = slim.conv2d(in_and_out15,     kernel_num * 16, [
                                3, 3], stride=1, scope='conv_15x15x512_1',     reuse=reuse)
                            print(in_and_out15)

                            in_and_out30 = slim.conv2d_transpose(in_and_out15,     kernel_num * 16, [
                                2, 2], stride=2, scope='deconv_30x30x512',   reuse=reuse)

                        in_and_out30 = tf.concat(
                            [in_and_out30, in30], 3, name='concat_30x30')
                        in_and_out30 = slim.dropout(
                            in_and_out30, keep_prob=dropout_keep_prob, scope='dropout')

                        with tf.variable_scope('out_30x30x256', reuse=reuse):
                            out30 = slim.conv2d(in_and_out30,   kernel_num * 8, [
                                3, 3], stride=1, scope='conv_30x30x256_1',   reuse=reuse)

                            if kwargs['unet_type'] == 'double_conv':
                                out30 = slim.conv2d(
                                    out30,          kernel_num * 8,
                                    [3, 3], stride=1, scope='conv_30x30x256_2',   reuse=reuse)

                            out60 = slim.conv2d_transpose(out30,        kernel_num * 8, [
                                2, 2], stride=2, scope='deconv_60x60x256',   reuse=reuse)
                        out60 = tf.concat(
                            [out60,   in60], 3, name='concat_60x60')
                        out60 = slim.dropout(
                            out60, keep_prob=dropout_keep_prob, scope='dropout')

                        with tf.variable_scope('out_60x60x128', reuse=reuse):
                            out60 = slim.conv2d(out60,          kernel_num * 4, [
                                3, 3], stride=1, scope='conv_60x60x128_1',   reuse=reuse)
                            if kwargs['unet_type'] == 'double_conv':
                                out60 = slim.conv2d(
                                    out60,          kernel_num * 4,
                                    [3, 3], stride=1, scope='conv_60x60x128_2',   reuse=reuse)
                            out120 = slim.conv2d_transpose(out60,        kernel_num * 4, [
                                2, 2], stride=2, scope='deconv_120x120x128', reuse=reuse)
                        out120 = tf.concat(
                            [out120, in120], 3, name='concat_120x120')
                        out120 = slim.dropout(
                            out120, keep_prob=dropout_keep_prob, scope='dropout')

                        with tf.variable_scope('out_120x120x64', reuse=reuse):
                            out120 = slim.conv2d(out120,          kernel_num * 2, [
                                3, 3], stride=1, scope='conv_120x120x64_1', reuse=reuse)
                            if kwargs['unet_type'] == 'double_conv':
                                out120 = slim.conv2d(
                                    out120,          kernel_num * 2,
                                    [3, 3], stride=1, scope='conv_120x120x64_2', reuse=reuse)
                            out240 = slim.conv2d_transpose(out120,        kernel_num * 2, [
                                2, 2], stride=2, scope='deconv_240x240x64', reuse=reuse)
                        # out240 = tf.concat(3, [out240, in240],
                        # name='concat_240x24032')
                        net = slim.dropout(
                            out240, keep_prob=dropout_keep_prob, scope='dropout')

                        with tf.variable_scope('out_240x240x32', reuse=reuse):
                            net = slim.conv2d(net,          kernel_num, [
                                3, 3], stride=1, scope='conv_240x240x32_1', reuse=reuse)
                            if kwargs['unet_type'] == 'double_conv':
                                net = slim.conv2d(
                                    net,          kernel_num, [3, 3],
                                    stride=1, scope='conv_240x240x32_2', reuse=reuse)

                        with tf.variable_scope('out_240x240x1', reuse=reuse):

                            num_output_channels = kwargs[
                                'label_shape'][-1]  # inputs.get_shape()[-1]

                            # add split if num_output_channels > 1:
                            if False:  # num_output_channels > 1:
                                net = slim.conv2d(net, kernel_num, [3, 3],
                                                  stride=1,
                                                  scope='out_conv1')

                                net = slim.conv2d(net, kernel_num, [3, 3],
                                                  stride=1,
                                                  scope='out_conv2')

                                print(
                                    "Added additional layers, outshape:", num_output_channels)

                            net = slim.conv2d(net, num_output_channels, [1, 1],
                                              stride=1,
                                              activation_fn=None,
                                              scope='conv_240x240x1')

                            net = slim.flatten(net, scope='flatten')
                            end_points['flatten'] = net

                            logits = net
                            end_points['logits'] = logits
                            end_points['predictions'] = tf.sigmoid(
                                logits, name='predictions')

        return logits, end_points

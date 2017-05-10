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

def classifier3D_graph(kwargs,
               inputs,
               dropout_keep_prob=0.8,
               kernel_num=32,
               num_classes=2,
               is_training=True,
               restore_logits=True,
               scope='classifier3D',
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
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}
    with tf.variable_scope(scope, [inputs], reuse=reuse):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.pool], stride=1, padding='SAME'):
                    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.relu):

                        shape = inputs.get_shape()
                        
                        #B,C,Z,Y,X,Ch
                        print_shape(inputs)
                        
                        inputs = tf.reshape(inputs, shape=[int(shape[0])*int(shape[1]), int(shape[2]), int(shape[3]), int(shape[4]), int(shape[5])])
                        print_shape(inputs)

                        #feature stage
                        #CONV
                        with tf.variable_scope('stage1', reuse=reuse):
                            net = slim.conv2d(inputs,         kernel_num , [
                                3, 3, 3], stride=1, scope='conv1',   reuse=reuse, data_format = 'NDHWC')       
                            #net = slim.conv2d(inputs,         kernel_num , [
                            #    3, 3, 3], stride=1, scope='conv2',   reuse=reuse, data_format = 'NDHWC')       
                            print_shape(net)

                        #Reduction
                        with tf.variable_scope('stage2', reuse=reuse):
                            net = slim.pool(
                                net,            [2, 2, 2], stride=2, scope='pool_32x32', data_format = 'NDHWC', pooling_type='MAX')
                            net = slim.conv2d(net,          kernel_num * 2, [
                                3, 3, 3], stride=1, scope='conv2',   reuse=reuse, data_format = 'NDHWC')
                            print_shape(net)


                        #Reduction
                        with tf.variable_scope('stage3', reuse=reuse):
                            net = slim.pool(
                                net,            [2, 2, 2], stride=2, scope='pool_16x16', data_format = 'NDHWC', pooling_type='MAX') 
                            net = slim.conv2d(net,          kernel_num * 3, [
                                    3, 3, 3], stride=1, scope='conv1',     reuse=reuse, data_format = 'NDHWC')
                            #net = slim.conv2d(net,         kernel_num * 2, [
                            #    3, 3, 3], stride=1, scope='conv2',   reuse=reuse, data_format = 'NDHWC')
                            #net = slim.conv2d(net,         kernel_num * 4, [
                            #    3, 3, 3], stride=1, scope='conv3',   reuse=reuse, data_format = 'NDHWC')
                            print_shape(net)


                        #Reduction
                        with tf.variable_scope('stage4', reuse=reuse):
                            net = slim.pool(
                                net,            [2, 2, 2], stride=2, scope='pool_8x8', data_format = 'NDHWC', pooling_type='MAX') 
                            net = slim.conv2d(net,          kernel_num * 4, [
                                    3, 3, 3], stride=1, scope='conv1',     reuse=reuse, data_format = 'NDHWC')
                            #net = slim.conv2d(net,         kernel_num * 4, [
                            #    3, 3, 3], stride=1, scope='conv2',   reuse=reuse, data_format = 'NDHWC')
                            #net = slim.conv2d(net,         kernel_num * 4, [
                            #    3, 3, 3], stride=1, scope='conv3',   reuse=reuse, data_format = 'NDHWC')
                            print_shape(net)
                        

                        #MIL Stage B*C, 8x8x8, kernels
                        with tf.variable_scope('MIL', reuse=reuse):
                            
                            feature_shape = net.get_shape()


                            net = slim.pool(net, [feature_shape[1], feature_shape[2], feature_shape[3]], stride=1,
                                                scope='MIL_candidate_pool', data_format = 'NDHWC', pooling_type='AVG', padding = 'VALID')
                            
                            print_shape(net)
                            
                            net = slim.flatten(net)
                            #B*C, ch

                            print_shape(net)

                            net = slim.fully_connected(net, 600, scope="fc1", activation_fn = tf.nn.relu)
                            net = slim.fully_connected(net, 600, scope="fc2", activation_fn = tf.nn.relu)
                            net = slim.fully_connected(net, 1, scope="fc3", activation_fn = None)
                            print_shape(net)


                            #use single candidate
                            #logits = net

                            #use all candidates
                            
                            net = tf.reshape(net, [int(shape[0]), int(shape[1])])
                            print_shape(net)
                            
                            #B, C

                            if 1:
                                logits = tf.reduce_mean(net, reduction_indices=[1])  
                                logits = tf.expand_dims(logits, 1)               
                            
                            print_shape(logits)

                        with tf.variable_scope('classifier', reuse=reuse):
                            #Bx1
                            end_points['logits'] = logits
                            end_points['predictions'] = tf.sigmoid(
                                logits, name='predictions')
                        
        return logits, end_points

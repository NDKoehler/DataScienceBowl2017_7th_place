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

from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import init_ops

from model_def import resnet2D_utils as resnet_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

import sys
import tensorflow as tf
slim = tf.contrib.slim

@add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with variable_scope.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = layers.batch_norm(
        inputs, activation_fn=nn_ops.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = layers_lib.conv2d(
          preact,
          depth, [1, 1],
          stride=stride,
          normalizer_fn=None,
          activation_fn=None,
          scope='shortcut')

    residual = layers_lib.conv2d(
        preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
    residual = resnet_utils.conv2d_same(
        residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
    residual = layers_lib.conv2d(
        residual,
        depth, [1, 1],
        stride=1,
        normalizer_fn=None,
        activation_fn=None,
        scope='conv3')

    output = shortcut + residual

    return utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):
  """Generator for v2 (preactivation) ResNet models.

  This function generates a family of ResNet v2 models. See the resnet_v2_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.


  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with variable_scope.variable_scope(
      scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with arg_scope(
        [layers_lib.conv2d, bottleneck, resnet_utils.stack_blocks_dense],
        outputs_collections=end_points_collection):
      net = inputs
      if include_root_block:
        if output_stride is not None:
          if output_stride % 4 != 0:
            raise ValueError('The output_stride needs to be a multiple of 4.')
          output_stride /= 4
        # We do not include batch normalization or activation functions in conv1
        # because the first ResNet unit will perform these. Cf. Appendix of [2].
        with arg_scope(
            [layers_lib.conv2d], activation_fn=None, normalizer_fn=None):
          net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
        net = layers.max_pool2d(net, [3, 3], stride=2, scope='pool1')
      net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
      # This is needed because the pre-activation variant does not have batch
      # normalization or activation functions in the residual unit output. See
      # Appendix of [2].
      net = layers.batch_norm(net, activation_fn=nn_ops.relu, scope='postnorm')
      if global_pool:
        # Global average pooling.
        net = math_ops.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
      if num_classes is not None:
        net = layers_lib.conv2d(
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='logits')
      # Convert end_points_collection into a dictionary of end_points.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if num_classes is not None:
        end_points['predictions'] = layers.softmax(net, scope='predictions')
      return net, end_points


resnet_v2.default_image_size = 224


def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_utils.Block('block1', bottleneck,
                         [(64, 32, 2)] * 2 ),
      resnet_utils.Block('block2', bottleneck,
                         [(128, 64, 1)] * 2 + [(128, 64, 2)]),
      resnet_utils.Block('block3', bottleneck,
                         [(256, 128, 1)] * 2 + [(256, 128, 2)]),
      #resnet_utils.Block('block4', bottleneck, [(160, 92, 1)] * 3)
  ]
  return resnet_v2(
      inputs,
      blocks,
      num_classes,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)


def print_shape(tensor):
    shape = tensor.get_shape()
    display_shape = [int(shape[x]) for x in range(len(shape))]
    print(tensor.name, display_shape)

def resnet2D_graph(kwargs,
               inputs,
               dropout_keep_prob=0.8,
               kernel_num=32,
               num_classes=2,
               is_training=True,
               restore_logits=True,
               scope='classifier2d',
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


                        plane_mil = kwargs['plane_mil']
                        shape = inputs.get_shape()
                        print_shape(inputs)

                        #permutate z dimension into last shape
                        if not plane_mil:
                          inputs = tf.transpose(inputs, [0,1,3,2,4,5])
                          print_shape(inputs)
                          inputs = tf.transpose(inputs, [0,1,2,4,3,5])                       
                          print_shape(inputs)
                        
                        shape_2 = inputs.get_shape()                      

                        
                        #Candidates into batch dimension, Z dimension into channel dimension for correct treatment for 2D convolution
                        if not plane_mil:
                          inputs = tf.reshape(inputs, [int(shape_2[0])*int(shape_2[1]),int(shape_2[2]),int(shape_2[3]),int(shape_2[4]) *int(shape_2[5]) ])                       
                        else:
                          inputs = tf.reshape(inputs, [int(shape_2[0])*int(shape_2[1])*int(shape_2[2]),int(shape_2[3]),int(shape_2[4]),int(shape_2[5]) ])                       
                        
                        #B,C,Z,Y,X,Ch
                        print_shape(inputs)
                        
                        num_classes = 1


                        '''
                        #--------------------------troll mode-----------------------------
                
                        with tf.variable_scope('in_240x240x32', reuse=reuse):
                            in240 = slim.conv2d(inputs,         kernel_num, [
                                3, 3], stride=1, scope='conv_240x240x32_0',   reuse=reuse)
                            in240 = slim.conv2d(
                                    in240,          kernel_num, [3, 3],
                                    stride=1, scope='conv_240x240x32_1',   reuse=reuse)
                        with tf.variable_scope('in_120x120x64', reuse=reuse):
                            in120 = slim.max_pool2d(
                                in240,            [2, 2], stride=2, scope='pool_120x120')
                            in120 = slim.conv2d(in120,          kernel_num * 2, [
                                3, 3], stride=1, scope='conv_120x120x64_0',   reuse=reuse)

                            in120 = slim.conv2d(
                                    in120,          kernel_num * 2,
                                    [3, 3], stride=1, scope='conv_120x120x64_1', reuse=reuse)
                        with tf.variable_scope('in_60x60x128', reuse=reuse):
                            in60 = slim.max_pool2d(
                                in120,            [2, 2], stride=2, scope='pool_60x60')
                            in60 = slim.conv2d(in60,          kernel_num * 4, [
                                3, 3], stride=1, scope='conv_60x60x128_0',     reuse=reuse)

                            in60 = slim.conv2d(
                                    in60,          kernel_num * 4,
                                    [3, 3], stride=1, scope='conv_60x60x128_1', reuse=reuse)

                        net = in60

                        net = math_ops.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

                        print_shape(net)


                        net = layers_lib.conv2d(
                            net,
                            num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            scope='logits')
                        print_shape(net)

                        
                        net = tf.reshape(in60, [kwargs['batch_size'], -1])
                        net = slim.fully_connected(net, 1000, scope='fc_out2')
                        net = slim.fully_connected(net, 1, scope='fc_out3', activation_fn = None)
                        logits = tf.sigmoid(net)


                        #Bx1
                        end_points['logits'] = logits
                        end_points['probs'] = logits

                        #--------------------------troll mode-----------------------------
                        '''
                

        
                        #resnet 50
                        out = resnet_v2_50(inputs, num_classes=num_classes, output_stride=8,  scope=scope, reuse=reuse)                       
                        print(out)

                        try:
                          net = out[-1]['resnet2D/resnet2D/logits']
                        except:
                          net = out[-1]['resnet2D_1/resnet2D/logits']
                                                      
                        #MIL Stage B*C, 8x8x8, kernels
                        with tf.variable_scope('MIL', reuse=reuse):
                                                        
                            if plane_mil:
                              net = tf.reshape(net, [int(shape[0]), int(shape[1]) * int(shape[2])])
                            else:
                              net = tf.reshape(net, [int(shape[0]), int(shape[1])])

                            print_shape(net)

                            #sigmoidal function to convert output layer of resnet to probabilities
                              #B, C


                            if 0:
                                net = tf.sigmoid(net)  

                                #NOISY AND mil function to reduce candidates to single predictions
                                b = variable_scope.get_variable('mil_param_a',
                                              shape=(num_classes,),
                                              initializer=init_ops.zeros_initializer(),
                                              regularizer=None,
                                              trainable=True,
                                              dtype=tf.float32)

                                print_shape(net)
                                p_mean = tf.reduce_mean(net, reduction_indices=[1])  
                                p_mean = tf.expand_dims(p_mean, 1)            

                                a = 10 #values from paper: 5,7.5,10
                                P = tf.sigmoid(a * (p_mean - b)) - tf.sigmoid(-a * b)
                                P = tf.divide(P, tf.sigmoid(a * (1 - b)) - tf.sigmoid(-a * b), name = 'probs')

                                with tf.variable_scope('classifier', reuse=reuse):
                                    #Bx1
                                    end_points['logits'] = p_mean
                                    end_points['probs'] = P
                                return p_mean, end_points
                            
                            #fc
                            if 0:
                                net = slim.fully_connected(net, 200, scope='fc_out1')
                                net = slim.fully_connected(net, 200, scope='fc_out2')
                                net = slim.fully_connected(net, 1, scope='fc_out3', activation_fn = None)
                                logits = tf.sigmoid(net)

                                with tf.variable_scope('classifier', reuse=reuse):
                                    #Bx1
                                    end_points['logits'] = logits
                                    end_points['probs'] = logits
                    

                            if 1:
                                #Maximum function for reducing candidates to single prediction
                                net = tf.sigmoid(net)  
                                print_shape(net)

                                logits = tf.reduce_max(net, reduction_indices=[1])  
                                logits = tf.expand_dims(logits, 1)            

                                print_shape(logits)

                                with tf.variable_scope('classifier', reuse=reuse):
                                    #Bx1
                                    end_points['logits'] = logits
                                    end_points['probs'] = logits

        return logits, end_points

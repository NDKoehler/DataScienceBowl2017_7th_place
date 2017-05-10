'''
Collection of Metrics used for Losses and Evaluation
'''
import sys
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops 
import cv2
import numpy as np



def MSE_segmentation(logits, labels, col_name, **H):
    logits = logits['logits'] 
    labels = labels['labels'] 

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')
    logits = tf.sigmoid(logits)
    
  
    loss = tf.reduce_mean(tf.square(logits - labels), name=col_name + '/MSE_segmentation')
    tf.add_to_collection(col_name, loss)

    return loss



def MSE_segmentation_separated_channels_metric(logits, labels, col_name, **H):
    logits = logits['logits'] 
    labels = labels['labels'] 

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')
    logits = tf.sigmoid(logits)

    labels_channels = tf.unstack(labels, axis=3)
    logits_channels = tf.unstack(logits, axis=3)
    total_loss = tf.constant(0,dtype=tf.float32)

    list_of_losses = []

    for cha in range(H['label_shape'][2]):
        #calc loss
        loss = tf.reduce_mean(tf.square(logits_channels[cha]- labels_channels[cha]), name=col_name + '/MSE'+str(cha))
        
        #add loss to collection (doesnt work since only last batch in val gets accounted)
        tf.add_to_collection(col_name+'_separated_'+str(cha), loss)
        
        #sum total loss
        total_loss = tf.add(loss, total_loss)

        #add to list
        list_of_losses.append(loss)

    total_loss = tf.div(total_loss,1.0, name=col_name+'MSE')
    list_of_losses.append(total_loss)


    tf.add_to_collection(col_name, total_loss)
    return list_of_losses

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = 1 - ( (intersection ) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth ) )
    return dice

def jaccard(y_true, y_pred):
    smooth = 1.
    intersection = tf.reduce_sum(y_true * y_pred)
    jaccard = 1 - ( (intersection + smooth  ) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + smooth ) )
    return jaccard

def jaccard_with_mse_block(y_true, y_pred):
    smooth = 1.
    intersection = tf.reduce_sum(y_true * y_pred)
    loss = tf.cond(tf.equal(tf.reduce_sum(y_true), tf.constant(0, dtype=tf.float32)), lambda: tf.reduce_mean(tf.square(y_true- y_pred)), lambda: 1 - ( (intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + smooth ) ))
    return loss

def dice(y_true, y_pred):
    smooth = 1.
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = 1 - ( (2 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth ) )
    return dice


def jaccard_separated_channels_metric(logits, labels, col_name, **H):
    logits = logits['logits'] 
    labels = labels['labels'] 

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')
    logits = tf.sigmoid(logits)

    labels_channels = tf.unstack(labels, axis=3)
    logits_channels = tf.unstack(logits, axis=3)
    total_loss = tf.constant(0,dtype=tf.float32)

    list_of_losses = []

    for cha in range(H['label_shape'][2]):
        #calc loss
        #loss = tf.reduce_mean(tf.square(logits_channels[cha]- labels_channels[cha]), name=col_name + '/MSE'+str(cha))
        loss = jaccard(labels_channels[cha], logits_channels[cha])

        #add loss to collection (doesnt work since only last batch in val gets accounted)
        tf.add_to_collection(col_name+'_separated_'+str(cha), loss)
        
        #sum total loss
        total_loss = tf.add(loss, total_loss)

         #add to list
        list_of_losses.append(loss)

    total_loss = tf.div(total_loss,1.0, name=col_name+'jaccard')
    list_of_losses.append(total_loss)

    tf.add_to_collection(col_name, total_loss)
    return list_of_losses

def jaccard_separated_channels(logits, labels, col_name, **H):
    logits = logits['logits']
    labels = labels['labels'] #first channel is label to predict, second channel is center

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')
    logits = tf.sigmoid(logits)

    labels_channels = tf.unstack(labels, axis=3)
    logits_channels = tf.unstack(logits, axis=3)
    total_loss = tf.constant(0,dtype=tf.float32)

    for cha in range(H['label_shape'][2]):
        #calc loss
        #loss = tf.reduce_mean(tf.square(logits_channels[cha]- labels_channels[cha]), name=col_name + '/MSE'+str(cha))
        loss = jaccard(labels_channels[cha], logits_channels[cha])

        #add loss to collection (doesnt work since only last batch in val gets accounted)
        tf.add_to_collection(col_name+'_separated_'+str(cha), loss)

        #sum total loss
        total_loss = tf.add(loss, total_loss)

        #add to list
        #list_of_losses.append(loss)

    total_loss = tf.div(total_loss,1.0, name=col_name+'jaccard')
    tf.add_to_collection(col_name, total_loss)
    return total_loss#, list_of_losses

def MSE_segmentation_separated_channels(logits, labels, col_name, **H):
    logits = logits['logits']
    labels = labels['labels']

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')
    logits = tf.sigmoid(logits)

    labels_channels = tf.unstack(labels, axis=3)
    logits_channels = tf.unstack(logits, axis=3)
    total_loss = tf.constant(0,dtype=tf.float32)

    #list_of_losses = []

    for cha in range(H['label_shape'][2]):
        #calc loss
        loss = tf.reduce_mean(tf.square(logits_channels[cha]- labels_channels[cha]), name=col_name + '/MSE'+str(cha))

        #add loss to collection (doesnt work since only last batch in val gets accounted)
        tf.add_to_collection(col_name+'_separated_'+str(cha), loss)

        #sum total loss
        total_loss = tf.add(loss, total_loss)

        #add to list
        #list_of_losses.append(loss)

    total_loss = tf.div(total_loss,1.0, name=col_name+'MSE')
    tf.add_to_collection(col_name, total_loss)
    return total_loss#, list_of_losses

class sensitivity():
    def __init__(self, logits, lab_mask2pred, lab_broaded, lab_center, min_prio, batch_size):
        super(sensitivity, self).__init__()

        self.logits = logits
        self.lab_broaded = lab_broaded
        self.lab_center = lab_center
        self.min_prio = min_prio
        self.batch_size = batch_size

        # inititalize values
        self.lab_center_min_prio_mask         = tf.cast(tf.greater(self.lab_center, self.min_prio), tf.float32)
        self.lab_not_in_broaded_min_prio_mask = tf.cast(tf.less_equal(self.lab_broaded, self.min_prio), tf.float32)
        self.lab_center_min_prio_mask_sum     = tf.cast(tf.reduce_sum(tf.reduce_sum(self.lab_center_min_prio_mask, axis=2),axis=1), tf.float32)

        self.logits_center_mask_sum     = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.greater(tf.multiply(self.logits, self.lab_center_min_prio_mask), self.min_prio), tf.float32), axis=2), axis=1)
        self.logits_not_in_broaded_mask = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.greater(tf.multiply(self.logits, self.lab_not_in_broaded_min_prio_mask), self.min_prio), tf.float32), axis=2), axis=1)
        self.mean_nodule_area           = tf.divide(tf.reduce_sum(lab_mask2pred), tf.reduce_sum(tf.cast(tf.greater(self.lab_center_min_prio_mask_sum,0), tf.float32)))

        self.true_positive_rate = 0

        self.update_true_positive_rate()
        self.update_false_positive_rate()


    def update_true_positive_rate(self):
        epsilon = 1e-6
        true_positive_rate = tf.cast(tf.divide(self.logits_center_mask_sum, epsilon+self.lab_center_min_prio_mask_sum), tf.float32)
        true_positive_rate = tf.multiply(true_positive_rate, tf.cast(tf.greater(self.lab_center_min_prio_mask_sum,0.), tf.float32)) # ensure no nans
        num_valid_data = tf.reduce_sum(tf.cast(tf.greater(self.lab_center_min_prio_mask_sum,0.),tf.float32))
        self.true_positive_rate = tf.where(tf.not_equal(num_valid_data,0), tf.reduce_sum( tf.divide(true_positive_rate, num_valid_data)), self.true_positive_rate)
    def get_true_positive_rate(self):
        return self.true_positive_rate

    def update_false_positive_rate(self):
        self.false_positive_rate = tf.reduce_mean(self.logits_not_in_broaded_mask/self.mean_nodule_area)

    def get_false_positive_rate(self):
        return self.false_positive_rate

def MSE_and_sensitivity_on_center(logits, labels, col_name, **H):
    logits = logits['logits']
    labels = labels['labels'] # first channel is mask to predict, second channel ist center
    lab_mask2pred, lab_center, lab_broaded = tf.split(labels, num_or_size_splits=3, axis=3, num=True)

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits        = tf.reshape(logits, size, name='reshape_logits')
    lab_mask2pred = tf.reshape(lab_mask2pred, size, name='reshape_lab_mask2pred')
    lab_center    = tf.reshape(lab_center, size, name='reshape_lab_center') 
    lab_broaded   = tf.reshape(lab_broaded, size, name='reshape_lab_center')

    logits = tf.sigmoid(logits)

    MSE_loss = tf.reduce_mean(tf.square(logits - lab_mask2pred), name=col_name + '/MSE')
    tf.add_to_collection(col_name, MSE_loss)

    min_prios = [0.2, 0.45, 0.7]
    # min_prios = [0.7]
    sensitivities = []
    for min_prio in min_prios:
        sensitivities.append(sensitivity(logits, lab_mask2pred, lab_broaded, lab_center, min_prio, H['batch_size']))

    true_positive_rates  = [sensi.get_true_positive_rate()  for sensi in sensitivities]
    false_positive_rates = [sensi.get_false_positive_rate() for sensi in sensitivities]
    # sensitivities_values = [sensi.get_sensitivities()       for sensi in sensitivities]
    for cnt, min_prio in enumerate(min_prios):
        tf.add_to_collection(col_name + '/true_positive_rate_'+str(min_prio),  tf.cast(true_positive_rates[cnt], tf.float32))
        tf.add_to_collection(col_name + '/false_positive_rate_'+str(min_prio), tf.cast(false_positive_rates[cnt], tf.float32))
        # tf.add_to_collection(col_name + '/sensitivity_'+str(min_prio),         tf.cast(sensitivities_values[cnt], tf.float32))

    return [MSE_loss] + false_positive_rates + true_positive_rates#, list_of_losses

def MSE_segmentation_with_center(logits, labels, col_name, **H):
    logits = logits['logits'] 
    labels = labels['labels'] 
    labels, _, _ = tf.split(labels, num_or_size_splits=3, axis=3, num=True)

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')
    logits = tf.sigmoid(logits)

  
    loss = tf.reduce_mean(tf.square(logits - labels), name=col_name + '/MSE_segmentation')
    tf.add_to_collection(col_name, loss)

    return loss

def MSE_weighted_segmentation_with_center(logits, labels, col_name, **H):
    logits = logits['logits'] 
    labels = labels['labels'] 
    labels, _, _ = tf.split(labels, num_or_size_splits=3, axis=3, num=True)

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')
    logits = tf.sigmoid(logits)

    weight_class1 = H['class1_weight']
  
    loss = tf.reduce_mean(weight_class1 * labels * tf.square(logits - labels) + (1-labels) * tf.square(logits - labels) , name=col_name + '/MSE_segmentation')
    tf.add_to_collection(col_name, loss)

    return loss

def logloss_with_center(logits, labels, col_name, **H):
    logits = logits['logits'] 
    labels = labels['labels'] 
    labels, _, _ = tf.split(labels, num_or_size_splits=3, axis=3, num=True)

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], 1)

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')
    logits = tf.sigmoid(logits)

    eps = 1e-6
    logits = tf.clip_by_value(logits, eps, 1.0-eps)

    weight = 10

    loss = tf.negative(weight*tf.reduce_mean(labels * tf.log(logits) + (1 - labels) * tf.log(1-logits)), name=col_name + '/logloss' )
    tf.add_to_collection(col_name, loss)
    return loss#, list_of_losses


def KL_with_center(logits, labels, col_name, **H):
    logits = logits['logits'] 
    labels = labels['labels'] 
    labels, _, _ = tf.split(labels, num_or_size_splits=3, axis=3, num=True)

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], 1)

    logits = tf.reshape(logits, size, name='reshape_logits')
    logits = tf.sigmoid(logits)
    labels = tf.reshape(labels, size, name='reshape_labels')

    eps = 1e-6
    logits = tf.clip_by_value(logits, eps, 1.0-eps)

    loss = tf.reduce_mean(labels * tf.log( tf.clip_by_value(labels, eps, 1.0-eps) / logits), name=col_name + '/KL' )

    tf.add_to_collection(col_name, loss)
    return loss#, list_of_losses

def jaccard_separated_channels_with_center_val(logits, labels, col_name, **H):
    logits = logits['logits']
    labels = labels['labels'] #first channel is label to predict, second channel is center
    labels, _, _ = tf.split(labels, num_or_size_splits=3, axis=3, num=True)

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')
    logits = tf.sigmoid(logits)

    labels_channels = tf.unstack(labels, axis=3)
    logits_channels = tf.unstack(logits, axis=3)
    total_loss = tf.constant(0,dtype=tf.float32)

    for cha in range(H['label_shape'][2]):
        #calc loss
        #loss = tf.reduce_mean(tf.square(logits_channels[cha]- labels_channels[cha]), name=col_name + '/MSE'+str(cha))
        loss = jaccard(labels_channels[cha], logits_channels[cha])

        #add loss to collection (doesnt work since only last batch in val gets accounted)
        tf.add_to_collection(col_name+'_separated_'+str(cha), loss)

        #sum total loss
        total_loss = tf.add(loss, total_loss)

        #add to list
        #list_of_losses.append(loss)

    total_loss = tf.div(total_loss,1.0, name=col_name+'jaccard')
    tf.add_to_collection(col_name, total_loss)
    return [total_loss]#, list_of_losses

def jaccard_separated_channels_with_center(logits, labels, col_name, **H):
    logits = logits['logits']
    labels = labels['labels'] #first channel is label to predict, second channel is center
    labels, _, _ = tf.split(labels, num_or_size_splits=3, axis=3, num=True)

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')
    logits = tf.sigmoid(logits)

    labels_channels = tf.unstack(labels, axis=3)
    logits_channels = tf.unstack(logits, axis=3)
    total_loss = tf.constant(0,dtype=tf.float32)

    for cha in range(H['label_shape'][2]):
        #calc loss
        #loss = tf.reduce_mean(tf.square(logits_channels[cha]- labels_channels[cha]), name=col_name + '/MSE'+str(cha))
        loss = jaccard(labels_channels[cha], logits_channels[cha])

        #add loss to collection (doesnt work since only last batch in val gets accounted)
        tf.add_to_collection(col_name+'_separated_'+str(cha), loss)

        #sum total loss
        total_loss = tf.add(loss, total_loss)

        #add to list
        #list_of_losses.append(loss)

    total_loss = tf.div(total_loss,1.0, name=col_name+'jaccard')
    tf.add_to_collection(col_name, total_loss)
    return total_loss#, list_of_losses

def add_loss_averages(losses, summary_name):
    # Compute the moving average of all individual losses and the total loss.
    #loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    #loss_averages_op = loss_averages.apply(losses)

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        raw = tf.summary.scalar(l.op.name +'(raw)', l)
        tf.add_to_collection(summary_name, raw)
        #avg = tf.scalar_summary(l.op.name, loss_averages.average(l))
        #tf.add_to_collection(summary_name, avg)
    return losses
  

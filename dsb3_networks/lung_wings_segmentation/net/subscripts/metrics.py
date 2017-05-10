'''
Collection of Metrics used for Losses and Evaluation
'''
import sys
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops 



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
    dice = 1 - ( (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth ) )
    return dice

def jaccard(y_true, y_pred):
    smooth = 1.
    intersection = tf.reduce_sum(y_true * y_pred)
    jaccard = 1 - ( (intersection  ) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + smooth ) )
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
    labels = labels['labels'] 

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

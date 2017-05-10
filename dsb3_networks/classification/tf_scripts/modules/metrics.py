'''
Collection of Metrics used for Losses and Evaluation
'''
import sys
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops 


def logloss(logits, labels, col_name, **H):
    logits = logits['probs'] 
    labels = labels['labels'] 

    eps = 1e-6
    logits = tf.clip_by_value(logits, eps, 1.0-eps)

    loss = tf.negative(tf.reduce_mean(labels * tf.log(logits) + (1 - labels) * tf.log(1-logits)), name=col_name + '/logloss' )
    tf.add_to_collection(col_name, loss)
    return loss#, list_of_losses


def MSE(logits, labels, col_name, **H):
    logits = logits['probs'] 
    labels = labels['labels'] 

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits - labels), 1),name=col_name + '/MSE')
    tf.add_to_collection(col_name, loss)
    return loss#loss


def logloss_weighted(logits, labels, col_name, **H):
    logits = logits['probs'] 
    labels = labels['labels'] 

    size = (H['batch_size'], H['label_shape'][0], H['label_shape'][1], H['label_shape'][2])

    logits = tf.reshape(logits, size, name='reshape_logits')
    labels = tf.reshape(labels, size, name='reshape_labels')

    weight = 100
    loss = tf.reduce_mean(labels * tf.log(logits) + weight*(1 - labels) * tf.log(1-logits), name=col_name + '/logloss'+str(cha) )
    tf.add_to_collection(col_name, loss)
    return loss#, list_of_losses


def logloss_rank(logits, labels, col_name, **H):
    logits = logits['probs'] 
    labels = labels['labels'] 

    eps = 1e-6
    logits = tf.clip_by_value(logits, eps, 1.0-eps)

    loss = tf.negative(tf.reduce_mean(labels * tf.log(logits) + (1 - labels) * tf.log(1-logits)), name=col_name + '/logloss_' + col_name + '/rank' )
    tf.add_to_collection(col_name, loss)
    return [loss, [logits, labels]]#, list_of_losses

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


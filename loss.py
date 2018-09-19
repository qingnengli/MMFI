import tensorflow as tf
import config
FLAGS = tf.app.flags.FLAGS

###############################################################
#                     Absolute Error                          #
###############################################################
def l1_loss(yp,gt,add_summary=True):
    l1_loss = tf.reduce_mean(tf.abs(gt - yp))
    if add_summary:
        tf.summary.scalar('L1_loss', l1_loss)
    return l1_loss
###############################################################
#                 Modified Absolute Error                     #
###############################################################
def mae_loss(yp,gt,add_summary=True):
    mae_loss = tf.reduce_mean(tf.log(1 + tf.exp(tf.abs(yp - gt))))
    if add_summary:
        tf.summary.scalar("MAE_loss", mae_loss)
    return mae_loss
###############################################################
#                       Binary Classification                 #
###############################################################
def dice_loss(yp,gt,add_summary=True):
    mask_front = gt
    mask_background = 1 - gt
    pro_front = yp
    pro_background = 1 - yp
    w1 = 1 / (tf.pow(tf.reduce_sum(mask_front), 2) + 1e-12)
    w2 = 1 / (tf.pow(tf.reduce_sum(mask_background), 2) + 1e-12)
    numerator = w1 * tf.reduce_sum(mask_front * pro_front) + w2 * tf.reduce_sum(mask_background * pro_background)
    denominator = w1 * tf.reduce_sum(mask_front + pro_front) + w2 * tf.reduce_sum(mask_background + pro_background)
    dice_loss = 1 - 2 * numerator / (denominator + 1e-12)
    if add_summary:
        tf.summary.scalar("Dice_loss", dice_loss)
    return dice_loss

def cross_entropy_loss(yp,gt,add_summary=True):
    mask_front = gt
    mask_background = 1 - gt
    pro_front = yp
    pro_background = 1 - yp
    mask = tf.ones_like(yp)
    w = (tf.reduce_sum(mask[:,:,:,0]) - tf.reduce_sum(yp)) / (tf.reduce_sum(yp) + 1e-12)
    cross_entropy_loss = -tf.reduce_mean(w * mask_front * tf.log(pro_front + 1e-12)
                                         + mask_background * tf.log(pro_background + 1e-12))
    if add_summary:
        tf.summary.scalar("Cross_entropy_loss", cross_entropy_loss)
    return cross_entropy_loss

def combine_loss(output, targets,add_summary=False,name=None):
  with tf.variable_scope('Combine_loss', reuse=True):
    l1 = l1_loss(output, targets,add_summary)
    mae = mae_loss(output, targets,add_summary)
    dice = dice_loss(output, targets,add_summary)
    entropy_loss = cross_entropy_loss(output, targets,add_summary)
    main_loss = l1 * FLAGS.l1_loss_weight\
                +mae * FLAGS.mae_loss_weight \
                + dice * FLAGS.dice_loss_weight \
                + entropy_loss * FLAGS.entropy_loss_weight
    if name is not None:
      tf.summary.scalar(name,main_loss)
  return main_loss

import tensorflow as tf
import config
FLAGS = tf.app.flags.FLAGS
###############################################################
#                PIX2PIX ADVERSARIAL LOSS                     #
###############################################################
def discriminator_loss(model,add_summaries=True):
  discrim_real_loss = tf.reduce_mean(tf.abs(1-(model.discriminator_real_outputs)))
  discrim_gen_loss = tf.reduce_mean(tf.abs(-1-(model.discriminator_gen_outputs)))
  discrim_loss = discrim_real_loss + discrim_gen_loss
  if add_summaries:
    tf.summary.scalar('discrim_real_loss',discrim_real_loss)
    tf.summary.scalar('discrim_gen_loss',discrim_gen_loss)
    tf.summary.scalar('discrim_loss', discrim_loss)
  return discrim_loss

def generator_loss(model,add_summaries=True):
  generator_loss = tf.reduce_mean(tf.abs(1-(model.discriminator_gen_outputs)))
  if add_summaries:
    tf.summary.scalar('generator_minimax_loss', generator_loss)
  return generator_loss


###############################################################
#                  Mean Absolute Error                        #
###############################################################
def l1_loss(yp,gt,add_summary=True):
    l1 = tf.reduce_mean(tf.abs(gt-yp))
    if add_summary:
        tf.summary.scalar('L1_loss', l1)
    return l1
###############################################################
#                 Modified Mean Absolute Error                #
###############################################################
def mmae_loss(yp,gt,add_summary=True):
    mmae_loss = tf.reduce_mean(tf.log(1 + tf.exp(tf.abs(yp - gt))))
    if add_summary:
        tf.summary.scalar("MMAE_loss", mmae_loss)
    return mmae_loss
###############################################################
#                       MS-SSIM Loss                          #
###############################################################
def SSIM_loss(yp,gt,add_summary=True):
    pred = tf.image.resize_images(yp,[256,256])
    label = tf.image.resize_images(gt,[256,256])
    ssim_loss = 1-tf.reduce_mean(tf.image.ssim_multiscale(pred,label,1.0))
    if add_summary:
        tf.summary.scalar("SSIM_loss", ssim_loss)
    return ssim_loss
###############################################################
#                       Binary Classification                 #
###############################################################
# def dice_loss(yp,gt,add_summary=True):
#     mask_front = gt
#     mask_background = 1 - gt
#     pro_front = yp
#     pro_background = 1 - yp
#     w1 = 1 / (tf.pow(tf.reduce_sum(mask_front), 2) + 1e-12)
#     w2 = 1 / (tf.pow(tf.reduce_sum(mask_background), 2) + 1e-12)
#     numerator = w1 * tf.reduce_sum(mask_front * pro_front) + w2 * tf.reduce_sum(mask_background * pro_background)
#     denominator = w1 * tf.reduce_sum(mask_front + pro_front) + w2 * tf.reduce_sum(mask_background + pro_background)
#     dice_loss = 1 - 2 * numerator / (denominator + 1e-12)
#     if add_summary:
#         tf.summary.scalar("Dice_loss", dice_loss)
#     return dice_loss
#
# def Dice_coe(yp, gt, thresh=0.5):
#   ypp = tf.cast(tf.greater(yp, thresh), dtype=tf.int32)
#   gtt = tf.cast(tf.greater(gt, thresh), dtype=tf.int32)
#   x = tf.reduce_sum(ypp * gtt)
#   y = tf.reduce_sum(ypp) + tf.reduce_sum(gtt)
#   return 2 * x / (tf.maximum(y, 1))
###############################################################
#                       Entropy Loss                          #
###############################################################
# def cross_entropy_loss(yp,gt,add_summary=True):
#     mask_front = gt
#     mask_background = 1 - gt
#     pro_front = yp
#     pro_background = 1 - yp
#     mask = tf.ones_like(yp)
#     w = (tf.reduce_sum(mask[:,:,:,0]) - tf.reduce_sum(yp)) / (tf.reduce_sum(yp) + 1e-12)
#     cross_entropy_loss = -tf.reduce_mean(w * mask_front * tf.log(pro_front + 1e-12)
#                                          + mask_background * tf.log(pro_background + 1e-12))
#     if add_summary:
#         tf.summary.scalar("Cross_entropy_loss", cross_entropy_loss)
#     return cross_entropy_loss

###############################################################
#                       Combination Loss                      #
###############################################################
def combine_loss(output, targets,add_summary=False,name=None):
  with tf.variable_scope('Combine_loss', reuse=True):
    L1 = l1_loss(output, targets,add_summary)
    mmae = mmae_loss(output, targets, add_summary)
    ssim = SSIM_loss(output, targets, add_summary)
    main_loss = L1 * FLAGS.l1_loss_weight + \
                mmae * FLAGS.mmae_loss_weight + \
                ssim * FLAGS.SSIM_loss_weight
    if name is not None:
      tf.summary.scalar(name,main_loss)
  return main_loss

###############################################################
#                       Evaluation Metrics                    #
###############################################################
def correlation(yp,gt):
  avg_x = tf.reduce_mean(yp)
  avg_y = tf.reduce_mean(gt)
  diffprod = tf.reduce_sum((yp-avg_x)*(gt-avg_y))
  xdiff2 = tf.reduce_sum(tf.squared_difference(yp, avg_x))
  ydiff2 = tf.reduce_sum(tf.squared_difference(gt, avg_y))
  return diffprod / tf.sqrt(xdiff2 * ydiff2)





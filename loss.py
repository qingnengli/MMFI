import numpy as np
import tensorflow as tf

###############################################################
#                       MS-SSIM                               #
###############################################################
def ssim_loss(yp,gt):
    ssim_loss = tf.reduce_mean(1 - tf_ms_ssim(gt, yp))
    tf.summary.scalar('SSIM_loss', ssim_loss)
    return ssim_loss
###############################################################
#                     Absolute Error                          #
###############################################################
def l1_loss(yp,gt):
    l1_loss = tf.reduce_mean(tf.abs(gt - yp))
    tf.summary.scalar('L1_loss', l1_loss)
    return l1_loss
###############################################################
#                 Modified Absolute Error                     #
###############################################################
def mae_loss(yp,gt):
    mae_loss = tf.reduce_mean(tf.log(1 + tf.exp(tf.abs(yp - gt))))
    tf.summary.scalar("MAE_loss", mae_loss)
    return mae_loss
###############################################################
#                       Binary Classification                 #
###############################################################
def dice_loss(yp,gt):
    mask_front = gt
    mask_background = 1 - gt
    pro_front = yp
    pro_background = 1 - yp
    w1 = 1 / (tf.pow(tf.reduce_sum(mask_front), 2) + 1e-12)
    w2 = 1 / (tf.pow(tf.reduce_sum(mask_background), 2) + 1e-12)
    numerator = w1 * tf.reduce_sum(mask_front * pro_front) + w2 * tf.reduce_sum(mask_background * pro_background)
    denominator = w1 * tf.reduce_sum(mask_front + pro_front) + w2 * tf.reduce_sum(mask_background + pro_background)
    dice_loss = 1 - 2 * numerator / (denominator + 1e-12)
    tf.summary.scalar("Dice_loss", dice_loss)
    return dice_loss

def cross_entropy_loss(yp,gt):
    mask_front = gt
    mask_background = 1 - gt
    pro_front = yp
    pro_background = 1 - yp
    mask = tf.ones_like(yp)
    w = (tf.reduce_sum(mask[:,:,:,0]) - tf.reduce_sum(yp)) / (tf.reduce_sum(yp) + 1e-12)
    cross_entropy_loss = -tf.reduce_mean(w * mask_front * tf.log(pro_front + 1e-12)
                                         + mask_background * tf.log(pro_background + 1e-12))
    tf.summary.scalar("Cross_entropy_loss", cross_entropy_loss)

    return cross_entropy_loss

###############################################################
#                        Function                             #
###############################################################
def _tf_fspecial_gauss(size, sigma, channels=1):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))

    window = g / tf.reduce_sum(g)
    return tf.tile(window, (1,1,channels,channels))

def tf_gauss_conv(img, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0
    window = _tf_fspecial_gauss(size, sigma, ch) # window shape [size, size]
    padded_img = tf.pad(img, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    return tf.nn.conv2d(padded_img, window, strides=[1,1,1,1], padding='VALID')

def tf_gauss_weighted_l1(img1, img2, mean_metric=True, filter_size=11, filter_sigma=1.5):
    diff = tf.abs(img1 - img2)
    l1 = tf_gauss_conv(diff, filter_size=filter_size, filter_sigma=filter_sigma)
    if mean_metric:
        return tf.reduce_mean(l1)
    else:
        return l1

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img1.get_shape().as_list()
    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0

    window = _tf_fspecial_gauss(size, sigma, ch) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    padded_img1 = tf.pad(img1, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    padded_img2 = tf.pad(img2, [[0, 0], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    mu1 = tf.nn.conv2d(padded_img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(padded_img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    paddedimg11 = padded_img1*padded_img1
    paddedimg22 = padded_img2*padded_img2
    paddedimg12 = padded_img1*padded_img2

    sigma1_sq = tf.nn.conv2d(paddedimg11, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(paddedimg22, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(paddedimg12, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    ssim_value = tf.clip_by_value(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)), 0, 1)
    if cs_map:
        cs_map_value = tf.clip_by_value((2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2), 0, 1)
        value = (ssim_value, cs_map_value)
    else:
        value = ssim_value
    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def tf_ms_ssim(img1, img2, weights=None, mean_metric=False):
    if weights is None:
        weights = [1, 1, 1, 1, 1] # [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] #[1, 1, 1, 1, 1] #
    level = len(weights)
    sigmas = [0.5]
    for i in range(level-1):
        sigmas.append(sigmas[-1]*2)
    weight = tf.constant(weights, dtype=tf.float32)
    mssim = []
    mcs = []
    for l, sigma in enumerate(sigmas):
        filter_size = int(max(sigma*4+1, 11))
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size, filter_sigma=sigma)
        mssim.append(ssim_map)
        mcs.append(cs_map)
    # list to tensor of dim D+1
    value = mssim[level-1]**weight[level-1]
    for l in range(level):
        value = value * (mcs[l]**weight[l])
    if mean_metric:
        return tf.reduce_mean(value)
    else:
        return value

def tf_ms_ssim_resize(img1, img2, weights=None, return_ssim_map=None, filter_size=11, filter_sigma=1.5):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    level = len(weights)
    assert return_ssim_map is None or return_ssim_map < level
    weight = tf.constant(weights, dtype=tf.float32)
    mssim = []
    mcs = []
    _, h, w, _ = img1.get_shape().as_list()
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False,
                                   filter_size=filter_size, filter_sigma=filter_sigma)
        if return_ssim_map == l:
            return_ssim_map = tf.image.resize_images(ssim_map, size=(h, w),
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        img1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*(mssim[level-1]**weight[level-1])
    if return_ssim_map is not None:
        return value, return_ssim_map
    else:
        return value

def tf_ssim_l1_loss(img1, img2, mean_metric=True, filter_size=11, filter_sigma=1.5, alpha=0.84):
    L1 = tf_gauss_weighted_l1(img1, img2, mean_metric=False, filter_size=filter_size, filter_sigma=filter_sigma)
    if mean_metric:
        loss_ssim= 1 - tf_ssim(img1, img2, cs_map=False, mean_metric=True,
                               filter_size=filter_size, filter_sigma=filter_sigma)
        loss_L1 = tf.reduce_mean(L1)
        value = loss_ssim * alpha + loss_L1 * (1-alpha)
    else:
        loss_ssim= 1 - tf_ssim(img1, img2, cs_map=False, mean_metric=False,
                               filter_size=filter_size, filter_sigma=filter_sigma)
        value = loss_ssim * alpha + L1 * (1-alpha)

    return value, loss_ssim

def tf_ms_ssim_l1_loss(img1, img2, mean_metric=True, alpha=0.84):
    ms_ssim_map = tf_ms_ssim(img1, img2, mean_metric=False)
    l1_map = tf_gauss_weighted_l1(img1, img2, mean_metric=False, filter_size=33, filter_sigma=8.0)
    loss_map = (1-ms_ssim_map) * alpha + l1_map * (1-alpha)
    if mean_metric:
        return tf.reduce_mean(loss_map)
    else:
        return loss_map

__author__ = 'PC-LiNing'

import tensorflow as tf

"""
    cnn_output=[batch_size,height,width,depth]
    norm_output=batch_norm(cnn_output)

    or
    cnn_output=[batch_size,total_num_filters]
    norm_output=batch_norm(tf.expand_dims(tf.expand_dims(cnn_output, 1), 1)
    cnn_output=tf.squeeze(norm_output)
"""


def batch_norm(x,epsilon=1e-5,ewma_decay=0.9,train=True):
    """ Batch normalization (cf. https://arxiv.org/abs/1502.03167).
         refence : http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
            norm(x) = 污(x-米)/考 +汕
            x = [batch,height,width,depth]
            米 = 米_B , batch mean
            考 = ﹟(考_B?+汍)
         note:
         apply batch normalization to the activation 考(Wx+b) would result in 考(BN(Wx+b)) , BN is the batch normalizing transform.
         train and test, the (mean,variance) used by BN is different.

    """
    shape = x.get_shape().as_list()
    depth = shape[-1]
    # scale = [depth] , init to 1
    # non-convolutional batch normalization, scale = [0]
    gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
    # offset = [depth] , init to 0
    # non-convolutional batch normalization, scale = [0]
    beta = tf.Variable(tf.constant(0.0, shape=[depth]))

    mean = tf.Variable(tf.constant(0.0, shape=[depth]),trainable=False)
    variance = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)
    ewma = tf.train.ExponentialMovingAverage(decay=ewma_decay)

    # convolutional batch normalization ㄛaxis = [0,1,2]
    axis = list(range(len(x.get_shape()) - 1))
    # non-convolutional batch normalization
    # axis = [0]

    if train:
        ema_apply_op = ewma.apply([mean, variance])
        mean, variance = tf.nn.moments(x, axis)
        assign_mean = mean.assign(mean)
        assign_variance = variance.assign(variance)
        with tf.control_dependencies([assign_mean, assign_variance]):
            return  tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon, scale_after_normalization=True)
    else:
        mean = ewma.average(mean)
        variance = ewma.average(variance)
        local_beta = tf.identity(beta)
        local_gamma = tf.identity(gamma)
        return tf.nn.batch_normalization(x, mean, variance, local_beta,local_gamma, epsilon, scale_after_normalization=True)
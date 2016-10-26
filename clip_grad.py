__author__ = 'PC-LiNing'

import tensorflow as tf

# example 1
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gvs = optimizer.compute_gradients(cost)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
optimizer.apply_gradients(capped_gvs)

# example 2
# max_grad_norm = 5
params = tf.trainable_variables()
grads = []
for grad in tf.gradients(loss, params):
  if grad is not None:
      // L2-norm clip
      grads.append(tf.clip_by_norm(grad, max_grad_norm))
  else:
      grads.append(grad)

global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer.apply_gradients(zip(grads, params), global_step=global_step)

# example 3
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gvs = optimizer.compute_gradients(cost)
capped_gvs = [(tf.clip_by_norm(grad,max_grad_norm), var) for grad, var in gvs]
optimizer.apply_gradients(capped_gvs)

# example 4
lr = 0.01
max_grad_norm = 5
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
opt = tf.train.GradientDescentOptimizer(lr)
optimizer.apply_gradients(zip(grads, tvars))
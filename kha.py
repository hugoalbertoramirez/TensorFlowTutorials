import tensorflow as tf

sess = tf.Session()

w = tf.Variable(2)
init = tf.global_variables_initializer()

sess.run(init)

print sess.run(w)

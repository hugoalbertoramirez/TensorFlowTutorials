import tensorflow as tf

sess = tf.Session()

# node1 = tf.constant(3.0, dtype=tf.float32)
# node2 = tf.constant(4.0) # also tf.float32 implicitly
# node3 = tf.add(node1, node2)
#
# print(sess.run(node3))
#
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a + b
#
# print(sess.run(adder_node, {a : 3, b : 5}))
# print(sess.run(adder_node, {a : [1, 1], b : [2, 3]}))
#
# add_triple = adder_node * 3
#
# print(sess.run(add_triple, {a : 3, b : 3}))

W = tf.Variable(.3)
b = tf.Variable(.3)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

#print(sess.run(linear_model, {x: [1, 2, 3]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

#print(sess.run(loss, {x: [1, 2, 3, 4], y : [1, 2, 3, 4]}))

fixW = tf.assign(W, -1)
fixb = tf.assign(b, 1)
sess.run([fixW, fixb])

#print(sess.run(loss, {x: [1, 2, 3, 4], y : [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)

for i in range(1500):
    sess.run(train, {x: [1, 2, 3, 4], y : [0, -1, -2, -3]})
    if i % 100 == 0:
        print i, sess.run([W, b, loss], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})



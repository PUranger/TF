import tensorflow as tf

#Computation Graph
x1 = tf.constant(5)
x2 = tf.constant(6)
#result = x1 * x2
result = tf.multiply(x1,x2)

print(result)

#sess = tf.Session()
#print(sess.run(result))
#sess.close()

#What supposed to happen in the session
with tf.Session() as sess:
	output = sess.run(result)
	print(output)

#in here, ok to have: print(output)
#but cannot have: sess.run(result)
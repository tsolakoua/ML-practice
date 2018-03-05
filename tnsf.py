import tensorflow as tf

x = tf.constant([1, 2, 3, 4])
y = tf.constant([5, 6, 7, 8])

result = tf.multiply(x, y)
print(result)

config = tf.ConfigProto(log_device_placement = True)


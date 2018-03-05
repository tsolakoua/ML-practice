import tensorflow as tf

T = 1.0
F = -1.0
bias = 1.0

train_input = [
	[T, T, bias],
	[T, F, bias],
	[F, T, bias],
	[F, F, bias],
]

# output for boolean OR operation
train_output = [
	[T],
	[T],
	[T],
	[F],
]

# weight tensor - 3 inputs correspond to 1 output
weight = tf.Variable(tf.random_normal([3, 1]))

# model
def step(x):
	is_greater = tf.greater(x,0)
	as_float = tf.to_float(is_greater)
	doubled = tf.multiply(as_float, 2)
	return tf.subtract(doubled, 1)

output = step(tf.matmul(train_input, weight))
error = tf.subtract(train_output, output)
mean_square_error = tf.reduce_mean(tf.square(error))

delta = tf.matmul(train_input, error, transpose_a=True)
train = tf.assign(weight, tf.add(weight, delta))

# init tensorflow session to evaluate the model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

err = 1
target = 0
epoch = 0
max_ep = 10

while epoch < max_ep and target < err:
	epoch = epoch + 1
	err, _ = sess.run([mean_square_error, train])
print('Epoch:', epoch, 'Mean Square Error:', err)

print(sess.run(output))
print(sess.run(weight))
sess.close()
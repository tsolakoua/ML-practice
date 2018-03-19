from numpy import array, random, exp, dot, transpose

input_train = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
output_train = array([[0, 1, 1, 0]]).T

weight = random.uniform(-1,1,(3,1))

def sigmoid(input_train, weight):
	return 1 / (1 + exp(-(dot(input_train, weight))))

def adjust_weight(input_train, output_train, out, weight):
	return dot(input_train.T, (output_train - out) * out * (1 - out))

for _ in range(1000):
	out = sigmoid(input_train, weight)
	weight = weight + adjust_weight(input_train, output_train, out, weight)
print (1 / (1 + exp(-(dot(array([1, 0, 0]), weight)))))

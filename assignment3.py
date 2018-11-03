from theano.tensor import *
from theano.tensor.nnet import *
import numpy

# x = fscalar('x')
# z = 3 * (x * x)
# derivative = grad(z, [x])
# differentiate = theano.function([x], derivative)
# print(differentiate(1))


x = fscalar('x')
y = fscalar('y')
w = fscalar('w')

z = -(x + y + w) ** 2

derivative = grad(z, [x, y])
differentiate = theano.function([x, y, w], derivative)
print(differentiate(0, 0, 1))


# v1 = numpy.asarray([1, 2])
# v2 = numpy.asarray([3, 4])
# print(numpy.dot(v1, v2))


# W = numpy.asarray([[1, 2],[3, 4],[5, 6]])
# v = numpy.asarray([1, 1])
# print(numpy.dot(W, v))


# texts = numpy.asanyarray([[1, 1], [2, 0], [1, 0], [0, 1], [0, 1]])
# scores = numpy.asanyarray([-2, 2, 0, 1, -1])
#
# W = theano.shared(numpy.asanyarray([0.0, 0.0]), 'W')
# predictions = dot(texts, W)
# Accuracy = -sqr(predictions - scores).sum() + 10
# gradients = grad(Accuracy, [W])
#
# W_updated = W + (0.1 * gradients[0])
# updates = [(W, W_updated)]
#
# f = theano.function([], Accuracy, updates=updates)
#
# for i in xrange(10):
#     output = f()
#     print(output)


# StartingPoint = numpy.asarray([0.0, 0.0])
# W = theano.shared(StartingPoint, 'W')
#
# z = -(W[0] + W[1] + 1) ** 2
# gradients = grad(z, [W])
#
# W_updated = W + (0.1 * gradients[0])
# updates = [(W, W_updated)]
#
# f = theano.function([], [W, z], updates=updates)
#
# for i in xrange(10):
#     output = f()
#     print(output)

# input_vector = fvector('input_vector')
# target_values = fvector('target_vector')
#
# W_initial_values = numpy.zeros((5, 2))
# W = theano.shared(W_initial_values, 'W')
# activations = dot(W, input_vector)
# predicted_values = sigmoid(activations)
# Accuracy = -sqr(predicted_values - target_values).sum()
# predicted_class = argmax(predicted_values)
# gradients = grad(Accuracy, W)
# updates = [(W, W + .1 * gradients)]
#
# train = theano.function([input_vector, target_values],
#                         [W, activations, predicted_values, predicted_class, Accuracy, gradients],
#                         updates=updates, allow_input_downcast=True)
#
# data_vector = [1., 0.]
# target_vector = [0, 0, 0, 0, 1]
# W, activations, predicted_values, predicted_class, Accuracy, gradients = train(data_vector, target_vector)
# print(W, activations, predicted_values, predicted_class, Accuracy)
# print(gradients)

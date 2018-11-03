from theano.tensor import *
from theano.tensor.nnet import *
import sys
import numpy
import collections

input_vector = theano.tensor.fvector('input_vector')  # theano variable representing image
target_values = theano.tensor.fvector('target_values')  # theano variable representing the label of that image

# This is just a template: it does not learn anything, and always returns the digit "0":
delta = 0.1

rng = numpy.random.RandomState(0)
# W1_initial_values = numpy.asarray(rng.normal(loc=0.0, scale=0.1, size=(500, 28 * 28)))
# W1 = theano.shared(W1_initial_values, 'W1')
#
# activations = dot(W1, input_vector)
# hidden_activations = sigmoid(activations)

W2_initial_values = numpy.asarray(rng.normal(loc=0.0, scale=0.1, size=(239, 28*28)))
W2 = theano.shared(W2_initial_values, 'W2')

activations = dot(W2, input_vector)
hidden_activations = sigmoid(activations)

W3_initial_values = numpy.asarray(rng.normal(loc=0.0, scale=0.1, size=(10, 239)))
W3 = theano.shared(W3_initial_values, 'W3')

predicted_activations = sigmoid(dot(W3, hidden_activations))

theano.config.on_unused_input = 'ignore'
predicted_class = argmax(predicted_activations)
Accuracy = -sqr(predicted_activations - target_values).sum()

gradients = grad(Accuracy, W3)
updates = [(W3, W3 + delta * gradients)]
# Change this to something meaningful and it will work!

# defining Theano functions for training and testing the model:
train = theano.function([input_vector, target_values], [Accuracy, predicted_class], updates=updates,
                        allow_input_downcast=True)
test = theano.function([input_vector, target_values], [Accuracy, predicted_class], allow_input_downcast=True)


# 'allow_input_downcast=True' is needed to avoid any issues converting between 64 and 32 bit numbers

def read_dataset(path):  # The function that reads training or testing data and returns it as an array of data points
    number_of_images = len(open(path).readlines()) / 28  # 28 lines per each image
    f = open(path)
    dataset = []  # starts with an empty container
    for i in range(int(number_of_images)):
        data_vector = []  # we start with empty data vector, and then we read the data for each image from the 28 lines that it takes:
        for l in range(28):  # each image takes 28 lines
            line = f.readline()
            line_parts = line.split()  # split the line into all the numbers
            assert (len(line_parts) == 29)  # should be total of 29: the label + 28 numbers for the image
            label = int(
                line_parts[0])  # very first number in the file is the label (the digit that the picture represents)
            assert (0 <= label <= 9)  # only digits 0-9 are allowed as labels
            # now, we will create "one-hot vector":
            label_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # a vector of 10 zeroes
            label_vector[label] = 1  # except 1 for the label
            data_vector += [float(line_parts[i]) / 255. for i in range(1, len(line_parts))]  # Python 3
            # data_vector += [float(line_parts[i])/255. for i in xrange(1, len(line_parts))]
            # we divide by 255 so the pixel brightness is represented by a number between 0 and 1

        dataset.append((label_vector, data_vector))
    return dataset


# reading the data:
data_test = read_dataset("test.txt")
data_train = read_dataset("train.txt")

# training
for epoch in range(10):
    cost_sum = 0.0
    correct = 0
    for labelv, vector in data_train:
        Accuracy, predicted_class = train(vector, labelv)
        cost_sum += Accuracy
        if (labelv[predicted_class] == 1):
            correct += 1
    print("Epoch: " + str(epoch) + ", Accuracy: " + str(cost_sum) + ", %correct: " + str(
        float(correct) / len(data_train) * 100))

# testing:
cost_sum = 0.0
correct = 0
for labelv, vector in data_test:
    Accuracy, predicted_class = test(vector, labelv)
    cost_sum += Accuracy
    if (labelv[predicted_class] == 1):
        correct += 1
print("\t%correct on the test set: " + str(float(correct) / len(data_test) * 100))

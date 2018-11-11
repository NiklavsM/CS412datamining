import numpy

# vals = numpy.zeros((5, 5))
#
# for i in range(5):
#     vals[i][i] = 1
#
#
# def map(words):
#     result = [0, 0, 0, 0, 0]
#     i = 0
#     while i < len(words):
#         result += vals[words[i]]
#         i += 1
#     return result
#
#
# print(map([0, 1, 2]))


# vals = numpy.zeros((5, 5))
# W_x = numpy.ones((5,5))
#
# for i in range(5):
#     vals[i][i] = 1
#
#
# def map(words):
#     result = [0, 0, 0, 0, 0]
#     i = 0
#     while i < len(words):
#         result += W_x.dot(vals[words[i]])
#         i += 1
#     return result
#
#
# print(map([0, 1, 2, 3]))


vals = numpy.zeros((5, 5))
W0 = numpy.ones((5, 5))
W1 = numpy.ones((5, 5))

for i in range(5):
    vals[i][i] = 1


def map(words):
    result = [0, 0, 0, 0, 0]
    i = 0
    while i < len(words):
        result += numpy.tanh(W0.dot(result) + W1.dot(vals[words[i]]))
        i += 1
    return result


print(map([0, 1, 2]))

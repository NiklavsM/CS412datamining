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



vals = numpy.zeros((5, 5))
W_x = numpy.ones((5,5))

for i in range(5):
    vals[i][i] = 1


def map(words):
    result = [0, 0, 0, 0, 0]
    i = 0
    while i < len(words):
        result += vals[words[i]] * W_x[words[i]]
        i += 1
    return result


print(map([0, 1, 2]))
import numpy

word_embedding_size = 5

vals = numpy.zeros((5, 5))
for i in range(5):
    vals[i][i] = 1

word_embeddings = vals
recurrent_size = 5

W_xh = numpy.ones((recurrent_size, word_embedding_size))
W_hh = numpy.ones((recurrent_size, recurrent_size))
W_xr = numpy.zeros((recurrent_size, word_embedding_size))
W_hr = numpy.zeros((recurrent_size, recurrent_size))
W_xz = numpy.zeros((recurrent_size, word_embedding_size))
W_hz = numpy.zeros((recurrent_size, recurrent_size))


def sigmoid(x):
    return 1. / (1. + numpy.exp(-x))


def rnn_step_tanh(x, h_prev):
    return numpy.tanh(numpy.dot(W_xh, x) + numpy.dot(W_hh, h_prev))


def map_sequence_add(input_indeces):
    input_vectors = word_embeddings[input_indeces]
    h = numpy.asarray([-1, 0, 0, 0, 0])
    for x in input_vectors:
        print("h: ", h, " x: ", x)
        h = rnn_step_tanh(x, h)
    print("final h: ", h)


map_sequence_add([0, 1, 2])

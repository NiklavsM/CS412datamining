from theano.tensor import *
import numpy

# word_embedding_size = 5
# input_indicies = ivector('input_indices')
#
# vals = numpy.zeros((5, 5))
# for i in range(5):
#     vals[i][i] = 1
# print("word_embeddings: \n", vals)
# word_embeddings = theano.shared(vals, 'word_embeddings')
# input_vectors = word_embeddings[input_indicies]
# recurrent_size = 5
#
#
# def rnn_step(x, h_prev):
#     return h_prev + x
#
#
# initial_context_vector = alloc(numpy.array(0, dtype=theano.config.floatX), recurrent_size)
# context_vector, other_info = theano.scan(
#     rnn_step,
#     sequences=input_vectors,
#     outputs_info=initial_context_vector,
#     non_sequences=[]
# )
# context_vector = context_vector[-1]
# map_sequence = theano.function([input_indicies], [input_vectors, context_vector], updates=[])
# vectors, context = map_sequence([0, 1, 0])
# print("Input vectors:\n", vectors, "\n are mapped into the context vector:\n", context)


word_embedding_size = 5
input_indicies = ivector('input_indices')

vals = numpy.zeros((5, 5))
for i in range(5):
    vals[i][i] = 1
# vals[4][4] = 5;
print("word_embeddings: \n", vals)
word_embeddings = theano.shared(vals, 'word_embeddings')
input_vectors = word_embeddings[input_indicies]
recurrent_size = 5
# W_x_Vals = numpy.zeros((5, 5))
# for i in range(5):
#     for k in range(5):
#          W_x_Vals[i][k] = 1
W_x = theano.shared(numpy.ones((5, 5)), 'W_x')


def rnn_step(x, h_prev, W_x):
    print("LOL w_x ", W_x.get_value())
    return h_prev + theano.tensor.dot(W_x, x)


initial_context_vector = alloc(numpy.array(0, dtype=theano.config.floatX), recurrent_size)
context_vector, other_info = theano.scan(
    rnn_step,
    sequences=input_vectors,
    outputs_info=initial_context_vector,
    non_sequences=[W_x]
)
print("context_vector", context_vector)
context_vector = context_vector[-1]
map_sequence = theano.function([input_indicies], [input_vectors, context_vector], updates=[])
vectors, context = map_sequence([0, 1, 2, 3])
print("Input vectors:\n", vectors, "\n are mapped into the context vector:\n", context)

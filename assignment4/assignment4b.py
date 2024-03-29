from theano.tensor import *
import numpy

word_embedding_size = 5
input_indicies = ivector('input_indices')

vals = numpy.zeros((5, 5))
for i in range(5):
    vals[i][i] = 1

print("word_embeddings: \n", vals)
word_embeddings = theano.shared(vals, 'word_embeddings')
input_vectors = word_embeddings[input_indicies]  ## ?
recurrent_size = 5

W_x = theano.shared(numpy.ones((5, 5)), 'W_x')

print("input_vectors ", word_embeddings.eval())


def rnn_step(x, h_prev, W_x):
    return h_prev + theano.tensor.dot(x, W_x)


initial_context_vector = alloc(numpy.array(0, dtype=theano.config.floatX), recurrent_size)
context_vector, other_info = theano.scan(
    rnn_step,
    sequences=input_vectors,
    outputs_info=initial_context_vector,
    non_sequences=[W_x]
)
print("context_vector", context_vector)
context_vector = context_vector[-1]
print("context_vector2", context_vector)
map_sequence = theano.function([input_indicies], [input_vectors, context_vector], updates=[])
vectors, context = map_sequence([0, 1, 2])
print("Input vectors:\n", vectors, "\n are mapped into the context vector:\n", context)

from theano.tensor import *
import numpy

word_embedding_size = 5
input_indicies = ivector('input_indices')

vals = numpy.zeros((5, 5))
for i in range(5):
    vals[i][i] = 1
# vals[4][4] = 5;
print("word_embeddings: \n", vals)
word_embeddings = theano.shared(vals, 'word_embeddings')
input_vectors = word_embeddings[input_indicies] ## ?
recurrent_size = 5
# W_x_Vals = numpy.zeros((5, 5))
# for i in range(5):
#     for k in range(5):
#          W_x_Vals[i][k] = 1
W0 = theano.shared(numpy.ones((5, 5)), 'W0')
W1 = theano.shared(numpy.ones((5, 5)), 'W1')

print("input_vectors " , word_embeddings.eval())
def rnn_step(x, h_prev, W0, W1):
    return theano.tensor.tanh(theano.tensor.dot(W0, h_prev) + theano.tensor.dot(W1, x))


initial_context_vector = alloc(numpy.array(0, dtype=theano.config.floatX), recurrent_size)
context_vector, other_info = theano.scan(
    rnn_step,
    sequences=input_vectors,
    outputs_info=initial_context_vector,
    non_sequences=[W0,W1]
)
print("context_vector", context_vector)
context_vector = context_vector[-1]
print("context_vector2", context_vector)
map_sequence = theano.function([input_indicies], [input_vectors, context_vector], updates=[])
vectors, context = map_sequence([0, 1, 2])
print("Input vectors:\n", vectors, "\n are mapped into the context vector:\n", context)
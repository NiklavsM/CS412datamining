import random
import theano
import collections
import numpy

# Total number of classes we need to map our sequences to.
n_classes = 200  # For example two classes: positive or negative (or true/false)
# We can change that later when needed
n_words = 200  # Total number of words we can support
recurrent_size = 50

# We represent the input sequence as a vector of integers (word id-s):
input_indices = theano.tensor.ivector('input_indices')
# We want to predict:
target_class = theano.tensor.iscalar('target_class')  # e.g. could be sentiment level
# All words in the language are represented as trainable vectors:
word_embedding_size = 50  # the size of those vectors
rng = numpy.random.RandomState(0)
vals = numpy.asarray(rng.normal(loc=0.0, scale=0.1, size=(n_words, word_embedding_size)))
word_embeddings = theano.shared(vals, 'word_embeddings')
# This represents the input sequence (e.g. a sentence):
input_vectors = word_embeddings[input_indices]

# This is just a template: it does not learn anything, and always returns the class "0":
W_o = theano.shared(numpy.asarray(rng.normal(loc=0.0,scale=0.1, size=(n_classes,recurrent_size)), dtype=theano.config.floatX), 'W_o')
initial_context_vector = theano.tensor.alloc(numpy.array(0, dtype=theano.config.floatX), recurrent_size)

W_xh = theano.shared(numpy.asarray(rng.normal(loc=0.0,scale=0.1, size=(recurrent_size,word_embedding_size)), dtype=theano.config.floatX), 'W_xh')
W_hh = theano.shared(numpy.asarray(rng.normal(loc=0.0,scale=0.1, size=(recurrent_size,recurrent_size)), dtype=theano.config.floatX), 'W_hh')
W_xr = theano.shared(numpy.asarray(rng.normal(loc=0.0,scale=0.1, size=(recurrent_size,word_embedding_size)), dtype=theano.config.floatX), 'W_xr')
W_hr = theano.shared(numpy.asarray(rng.normal(loc=0.0,scale=0.1, size=(recurrent_size,recurrent_size)), dtype=theano.config.floatX), 'W_hr')
W_xz = theano.shared(numpy.asarray(rng.normal(loc=0.0,scale=0.1, size=(recurrent_size,word_embedding_size)), dtype=theano.config.floatX), 'W_xz')
W_hz = theano.shared(numpy.asarray(rng.normal(loc=0.0,scale=0.1, size=(recurrent_size,recurrent_size)), dtype=theano.config.floatX), 'W_hz')
def rnn_step(x, h_prev):
    r = theano.tensor.nnet.nnet.sigmoid(theano.tensor.dot(W_xr, x) + theano.tensor.dot(W_hr, h_prev))
    z = theano.tensor.nnet.nnet.sigmoid(theano.tensor.dot(W_xz, x) + theano.tensor.dot(W_hz, h_prev))
    _h = theano.tensor.tanh(theano.tensor.dot(W_xh, x) + theano.tensor.dot(theano.tensor.dot(W_hh, r), h_prev))
    return z * h_prev + (1.0 - z) * _h


context_vector, other_info = theano.scan(
    rnn_step,
    sequences = input_vectors,
    outputs_info = initial_context_vector,
    non_sequences = []
)
context_vector = context_vector[-1]
activations = theano.tensor.dot(W_o, context_vector)


output = theano.tensor.nnet.softmax([activations])[0]
predicted_class = theano.tensor.argmax(output)
cost = -theano.tensor.log(output[target_class])
updates = [
    (word_embeddings,word_embeddings - .01*theano.tensor.grad(cost,word_embeddings)),
	(W_o, W_o - .01*theano.tensor.grad(cost,W_o)),
    (W_xh, W_xh - .01*theano.tensor.grad(cost,W_xh)),
    (W_hh, W_hh - .01*theano.tensor.grad(cost,W_hh)),
	(W_xr, W_xr - .01*theano.tensor.grad(cost,W_xr)),
    (W_hr, W_hr - .01*theano.tensor.grad(cost,W_hr)),
	(W_xz, W_xz - .01*theano.tensor.grad(cost,W_xz)),
    (W_hz, W_hz - .01*theano.tensor.grad(cost,W_hz))
]
theano.config.on_unused_input = 'ignore'
Accuracy = cost

# Change this to something meaningful and it will work!

train = theano.function([input_indices, target_class], [Accuracy, predicted_class], updates=updates,
                        allow_input_downcast=True)
test = theano.function([input_indices, target_class], [Accuracy, predicted_class], allow_input_downcast=True)


def read_dataset(
        path):  # reading text data in the simple format: integer score, followed by a tab, followed by the text sequence
    dataset = []
    with open(path, "r") as f:
        for line in f:
            line_parts = line.strip().split("\t")
            if (len(line_parts) > 1):  # convering to low case and adding spaces between any punctuation:
                s1 = line_parts[1].lower().replace(',', ' , ').replace(';', ' ; ').replace(':', ' : ').replace('"',
                                                                                                               ' " ').replace(
                    "'", " ' ").replace("-", " - ").replace("(", " ( ").replace(")", " ) ").replace(
                    "?", " ? ").replace("!", " ! ").replace(".", " . ")
                dataset.append((line_parts[0], s1))
    return dataset


def create_dictionary(sentences):  # this is indexing! we need to convert all words to their ID-s
    counter = collections.Counter()  # Python's class that can count
    for sentence in sentences:
        for word in sentence:
            counter.update([word])

    word2id = collections.OrderedDict()  # Python's class that can map words to ID-s
    word2id["<unk>"] = 0  # We reserve this for "uknown words" that we may encounter in the future
    word2id["<s>"] = 1  # Marks beginning of the sentence
    word2id["</s>"] = 2  # Marks the end of the sentence

    word_count_list = counter.most_common()  # For every word, we create an entry in  'word2id'
    for (word, count) in word_count_list:  # so it can map them to their ID-s
        word2id[word] = len(word2id)

    return word2id


def sentence2ids(words, word2id):  # Converts a word sequence (sentence) into a list of ID-s
    ids = [word2id["<s>"], ]  # marks beginning of the sentence
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(word2id["<unk>"])
    # ids.append(word2id["</s>"])  # marks the end of the sentence
    return ids

def word2ids(word, word2id):
    if word in word2id:
        return word2id[word]
    else:
        return word2id["<unk>"]

path_train = "data/testTrainData.txt"
path_test = "data/testTrainData.txt"

sentences_train = read_dataset(path_train)
sentences_test = read_dataset(path_test)
word2id = create_dictionary([sentence.split() for label, sentence in sentences_train])
# n_words = len(word2id)  # Important to set it c
data_train = [(word2id[label], sentence2ids(sentence.split(), word2id)) for label, sentence in sentences_train]  # here we need to convert
data_test = [(word2id[label], sentence2ids(sentence.split(), word2id)) for label, sentence in sentences_test]  # our data from text to the lists of ID-s


def generate_sentence(start):
    count = 0
    s = "<s> " + start + " "
    sv = sentence2ids(s.split(), word2id)
    while True:
        cost, predicted_class = test(sv, 1)
        count += 1
        if predicted_class == word2id["</s>"] or count > 100:
            break
        else:
            if predicted_class < len(word2id):
                word = list(word2id.items())[predicted_class][0]
                sv.append(predicted_class)
            else:
                word = "<unk>"
                sv.append(word2id["<unk>"])
            s += word + " "
    print(s)


# The rest of the code is similar to the MNIST task:

for epoch in range(40):

    cost_sum = 0.0
    correct = 0
    count = 0
    for target_class, sentence in data_train:
        count += 1
        cost, predicted_class = train(sentence, target_class)
        cost_sum += cost
        if predicted_class == target_class:
            correct += 1
    print("Epoch: " + str(epoch) + "\tCost: " + str(cost_sum) + "\tAccuracy: " + str(float(correct) / count))
    cost_sum2 = 0.0
    correct2 = 0
    for target_class, sentence in data_test:
        cost, predicted_class = test(sentence, target_class)
        cost_sum2 += cost
        if predicted_class == target_class:
            correct2 += 1
    print("\t\t\t\t\t\t\tTest_cost: " + str(cost_sum2) + "\tTest_accuracy: " + str(float(correct2) / len(data_test)))
    generate_sentence("there is")
    generate_sentence("there are")
    generate_sentence("the")
    generate_sentence("at least")
    generate_sentence("there is 1")

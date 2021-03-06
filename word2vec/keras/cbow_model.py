from __future__ import absolute_import

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import plot_model
import math
import numpy as np
import tensorflow as tf
#from tensorflow.python.keras.utils.np_utils import accuracy
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Lambda, Dense
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.optimizers import SGD
#from tensorflow.python.keras.objectives import mse
from word2vec.keras import global_settings as G
from word2vec.keras.sentences_generator import Sentences
from word2vec.keras import vocab_generator as V_gen
from word2vec.keras import save_embeddings as S
from tensorflow.python.keras.layers import Dot
# import global_settings as G
# from sentences_generator import Sentences
# import vocab_generator as V_gen
# import save_embeddings as S

k = G.window_size # context windows size
context_size = 2*k

# Creating a sentence generator from demo file
sentences = Sentences("test_file.txt")

vocabulary = dict()
V_gen.build_vocabulary(vocabulary, sentences)
V_gen.filter_vocabulary_based_on(vocabulary, G.min_count)
reverse_vocabulary = V_gen.generate_inverse_vocabulary_lookup(vocabulary, "vocab.txt")

# generate embedding matrix with all values between -1/2d, 1/2d
import os
embedding = np.ndarray
aFile = "initialShare.npy"
if os.path.isfile(aFile):
    embedding = np.load(aFile)
else:
    embedding = np.random.uniform(-1.0/2.0/G.embedding_dimension, 1.0/2.0/G.embedding_dimension, (G.vocab_size, G.embedding_dimension))
    np.save(aFile, embedding)

embeddingTwo = np.zeros((G.vocab_size, G.embedding_dimension))
# Creating CBOW model
# Model has 3 inputs
# Current word index, context words indexes and negative sampled word indexes
word_index = Input(shape=(1,), name="word")
context = Input(shape=(context_size,), name="context")
negative_samples = Input(shape=(G.vocab_size - 1,), name="negative")
# All the inputs are processed through a common embedding layer
shared_embedding_layer = Embedding(input_dim=(G.vocab_size), output_dim=G.embedding_dimension, weights=[embedding])
shared_embedding_layer2 = Embedding(input_dim=(G.vocab_size), output_dim=G.embedding_dimension, weights=[embeddingTwo])

word_embedding = shared_embedding_layer(word_index)
word_embedding = Lambda(lambda x: x * 1)(word_embedding)
context_embeddings = shared_embedding_layer2(context)
negative_words_embedding = shared_embedding_layer(negative_samples)
negative_words_embedding = Lambda(lambda x: x * 1)(negative_words_embedding)

# Now the context words are averaged to get the CBOW vector
cbow = Lambda(lambda x: K.mean(x, axis=1), output_shape=(G.embedding_dimension,))(context_embeddings)
# The context is multiplied (dot product) with current word and negative sampled words
print(type(word_embedding))
print(type(cbow))
word_context_product = Dot(axes=-1)([word_embedding, cbow])
word_context_product = Lambda(lambda x: tf.math.sigmoid(x))(word_context_product)

# word_context_product = Dense(1,activation = "sigmoid")(word_context_product)
print(K.shape(word_embedding))
print(K.shape(word_context_product))
print(K.shape(cbow))
negative_context_product = Dot(axes=-1)([negative_words_embedding, cbow])
# negative_context_product = Dense(1, activation = "sigmoid")(negative_context_product)
boost = 1
import sys
if len(sys.argv)>5:
    boost = float(sys.argv[5])
if boost > 1:
    negative_context_product = Lambda(lambda x: x * boost)(negative_context_product)
negative_context_product = Lambda(lambda x: tf.math.sigmoid(x))(negative_context_product)
# The dot products are outputted

model = Model(inputs=[word_index, context, negative_samples], outputs=[word_context_product, negative_context_product])
# binary crossentropy is applied on the output
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
print (model.summary())
plot_model(model, to_file='model.png')
print(V_gen.getStepsPerEpoch(sentences, batchSize=1))
# model.fit_generator(V_gen.pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary), samples_per_epoch=G.train_words, nb_epoch=1)
test = next(V_gen.pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary))
model.fit_generator(V_gen.pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary),epochs=1, steps_per_epoch=V_gen.getStepsPerEpoch(sentences, batchSize=1))
# Save the trained embedding

# outputs = [layer.output for layer in model.layers]
# functors = [K.function([model.input, K.learning_phase()], [out]) for out in outputs]
#
# layer_outs = [func([test[0], 1.]) for func in functors]
print(test)

layer_name = 'lambda'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(test[0])[0]

layer_name = 'lambda_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output_1 = intermediate_layer_model.predict(test[0])[0]

layer_name = 'lambda_2'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output_2 = intermediate_layer_model.predict(test[0])[0]


layer_name = 'dot'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
dot = intermediate_layer_model.predict(test[0])[0]


layer_name = 'dot_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
dot1 = intermediate_layer_model.predict(test[0])[0]

# get_3rd_layer_output = K.function([model.layers[0].input],
#                                   [model.layers[3].output])
# layer_output = get_3rd_layer_output([word_index[1], context[1], negative_samples[1]])[0]

S.save_embeddings("embedding.txt", shared_embedding_layer.get_weights()[0], vocabulary)
S.save_embeddings("embedding2.txt", shared_embedding_layer2.get_weights()[0], vocabulary)
S.save_embeddings_binary("embeddingb", shared_embedding_layer.get_weights()[0], vocabulary)
S.save_embeddings_binary("embeddingb2", shared_embedding_layer2.get_weights()[0], vocabulary)
# S.save_embeddings("embedding.txt", shared_embedding_layer.get_weights()[1], vocabulary)

# input_context = np.random.randint(10, size=(1, context_size))
# input_word = np.random.randint(10, size=(1,))
# input_negative = np.random.randint(10, size=(1, G.negative))

# print "word, context, negative samples"
# print input_word.shape, input_word
# print input_context.shape, input_context
# print input_negative.shape, input_negative

# output_dot_product, output_negative_product = model.predict([input_word, input_context, input_negative])
# print "word cbow dot product"
# print output_dot_product.shape, output_dot_product
# print "cbow negative dot product"
# print output_negative_product.shape, output_negative_product

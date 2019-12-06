#we give encoder input sequence like 'hello how are you', we take the last hidden state and feed to decoder and it
#will generate a decoded value. we compare that to target value, if translation would be 'bonjour ca va' and minimize
#the difference by optimizing a loss function

#in this case we just want to encode and decode the input successfully

#bidirectional encoder
#We will teach our model to memorize and reproduce input sequence.
#Sequences will be random, with varying length.
#Since random sequences do not contain any structure,
#model will not be able to exploit any patterns in data.
#It will simply encode sequence in a thought vector, then decode from it.
#this is not about prediction (end goal), it's about understanding this architecture

#this is an encoder-decoder architecture. The encoder is bidrectional so
#it It feeds previously generated tokens during training as inputs, instead of target sequence.

import numpy as np #matrix math
import tensorflow as tf #machine learningt
# import helpers #for formatting data into batches and generating random sequence data

tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session

tf.__version__

#First critical thing to decide: vocabulary size.
#Dynamic RNN models can be adapted to different batch sizes
#and sequence lengths without retraining
#(e.g. by serializing model parameters and Graph definitions via tf.train.Saver),
#but changing vocabulary size requires retraining the model.

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20 #character length

encoder_hidden_units = 20 #num neurons
decoder_hidden_units = encoder_hidden_units * 2 #in original paper, they used same number of neurons for both encoder
#and decoder, but we use twice as many so decoded output is different, the target value is the original input
#in this example

#input placehodlers
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
#contains the lengths for each of the sequence in the batch, we will pad so all the same
#if you don't want to pad, check out dynamic memory networks to input variable length sequences
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

#randomly initialized embedding matrrix that can fit input sequence
#used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
#reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

#this thing could get huge in a real world application
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple

encoder_cell = LSTMCell(encoder_hidden_units)


#get outputs and states
#bidirectional RNN function takes a separate cell argument for
#both the forward and backward RNN, and returns separate
#outputs and states for both the forward and backward RNN

#When using a standard RNN to make predictions we are only taking the “past” into account.
#For certain tasks this makes sense (e.g. predicting the next word), but for some tasks
#it would be useful to take both the past and the future into account. Think of a tagging task,
#like part-of-speech tagging, where we want to assign a tag to each word in a sentence.
#Here we already know the full sequence of words, and for each word we want to take not only the
#words to the left (past) but also the words to the right (future) into account when making a prediction.
#Bidirectional RNNs do exactly that. A bidirectional RNN is a combination of two RNNs – one runs forward from
#“left to right” and one runs backward from “right to left”. These are commonly used for tagging tasks, or
#when we want to embed a sequence into a fixed-length vector (beyond the scope of this post).


((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float64, time_major=True)
    )

encoder_fw_outputs
encoder_bw_outputs
encoder_fw_final_state
encoder_bw_final_state


#Concatenates tensors along one dimension.
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

#letters h and c are commonly used to denote "output value" and "cell state".
#http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#Those tensors represent combined internal state of the cell, and should be passed together.

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

#TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)
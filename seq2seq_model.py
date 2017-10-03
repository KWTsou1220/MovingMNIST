import tensorflow as tf
from seq2seq_model import *

def seq2seq(encoder_cell, decoder_cell, encoder_input, decoder_input, init_state, tensor, feed_previous=False, scope=None):
    """
    encoder_cell: a list of tf.nn.rnn_cell object
    decoder_cell: a list of tf.nn.rnn_cell object
    encoder_input: [time_steps, batch_size, input_size]
    decoder_input: [time_steps, batch_size, input_size]
    init_state: a list of tensor [batch_size, state_size]
    feed_previous: use preivous output or not
    """
    
    # Encoder
    _, state = Encoder(encoder_cell=encoder_cell, encoder_input=encoder_input, init_state=init_state, scope="Encoder")
    # Decoder
    output, state = Decoder(decoder_cell=decoder_cell, decoder_input=decoder_input, last_encoder_input=encoder_input[-1, :, :], 
                            init_state=state, tensor=tensor, feed_previous=feed_previous, scope="Decoder")
    return output
    
        
        
def Encoder(encoder_cell, encoder_input, init_state, scope="Encoder"):
    """
    encoder_cell: a list of tf.nn.rnn_cell object
    """
    Depth = len(encoder_cell)
    Time_step = encoder_input.get_shape().as_list()[0]
    state = init_state
    hidden = []
    with tf.variable_scope(scope or "Encoder", reuse=True):
        for time_step in xrange(Time_step):
            if time_step != 0:
                for depth in xrange(Depth):
                    with tf.variable_scope("layer"+str(depth), reuse=True):
                        if depth != 0:
                            hidd, state[depth] = encoder_cell[depth](hidden[depth-1][time_step, :, :], state[depth])
                        else:
                            hidd, state[depth] = encoder_cell[depth](encoder_input[time_step, :, :], state[depth])
                        hidd = tf.expand_dims(hidd, axis=0)
                        hidden[depth] = tf.concat(concat_dim=0, values=[hidden[depth], hidd]) # [time_step, batch_size, input_size]
            else:
                # hidden: a list of length Depth and contains tensors [batch_size, input_size]
                for depth in xrange(Depth):
                    with tf.variable_scope("layer"+str(depth), reuse=True):
                        if depth != 0:
                            hidd, state[depth] = encoder_cell[depth](hidden[depth-1][time_step, :, :], state[depth])
                        else:
                            hidd, state[depth] = encoder_cell[depth](encoder_input[time_step, :, :], state[depth])
                        hidd = tf.expand_dims(hidd, axis=0)
                        hidden.append(hidd)
    # hidden: a list of length Depth and contains tensor of shape [time_step, batch_size, input_size]
    return hidden, state # final state
    
        
def Decoder(decoder_cell, decoder_input, last_encoder_input, init_state, tensor, feed_previous=False, scope="Decoder"):
    """
    decoder_cell: a list of tf.nn.rnn_cell object
    tensor: convolution or not
    last_layer_shape: the shape of last seq2seq layer
    output_shape: the shape of the target
    """
    Time_step, Batch_size = decoder_input.get_shape().as_list()[0:2]
    Output_shape = decoder_input.get_shape().as_list()[2:] # list of input_shape: if tensor: [height, width, ch], else: [num_units]
    Depth = len(decoder_cell)
    state = init_state
    hidden = []
    with tf.variable_scope(scope or "Decoder", reuse=True):
        for time_step in xrange(Time_step):
            if time_step != 0:
                for depth in xrange(Depth):
                    with tf.variable_scope("layer"+str(depth), reuse=True):
                        if depth == 0:
                            if feed_previous == False:
                                hidd, state[depth] = decoder_cell[depth](decoder_input[time_step-1, :, :], state[depth])
                            else:
                                hidd, state[depth] = decoder_cell[depth](tf.sigmoid(hidden[-1][time_step-1, :, :]), state[depth])
                        elif depth != 0:
                            hidd, state[depth] = decoder_cell[depth](hidden[depth-1][time_step, :, :], state[depth])
                        if depth == Depth-1: # last layer
                            hidd = output_layer(hidd, decoder_cell[depth], tensor, Output_shape)
                        hidd = tf.expand_dims(hidd, axis=0)
                        hidden[depth] = tf.concat(concat_dim=0, values=[hidden[depth], hidd]) # [time_step, batch_size, input_size]
            else:
                # hidden: a list of length Depth and contains tensors [batch_size, input_size]
                for depth in xrange(Depth):
                    with tf.variable_scope("layer"+str(depth), reuse=True):
                        if depth == 0: # first layer and first time step
                            hidd, state[depth] = decoder_cell[depth](last_encoder_input, state[depth])
                        elif depth != 0:
                            hidd, state[depth] = decoder_cell[depth](hidden[depth-1][time_step, :, :], state[depth])
                        if depth == Depth-1: # last layer
                            hidd = output_layer(hidd, decoder_cell[depth], tensor, Output_shape)
                        hidd = tf.expand_dims(hidd, axis=0)
                        hidden.append(hidd)
    # hidden: a list of length Depth and contains tensor of shape [time_step, batch_size, input_size]
    
    return hidden[-1], state
    
def output_layer(hidd, decoder_cell, tensor, output_shape):
    if tensor == True:
        last_layer_shape = [decoder_cell.batch_size, decoder_cell.output_height, decoder_cell.output_width,
                                                    decoder_cell.output_ch]
        hidd = tf.reshape(hidd, shape=last_layer_shape)
        W = tf.get_variable("final_weights")
        b = tf.get_variable("final_biases")
        hidd = tf.nn.conv2d(hidd, filter=W, strides=[1, 1, 1, 1], padding="SAME")
        hidd = tmp = tf.nn.bias_add(hidd, b)
        hidd = tf.reshape(hidd, shape=output_shape)
    else:
        W = tf.get_variable("final_weights")
        b = tf.get_variable("final_biases")
        hidd = tf.nn.bias_add(tf.matmul(hidd, W), b)
    return hidd # [batch_size, input_size]

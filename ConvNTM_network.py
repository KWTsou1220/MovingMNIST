import tensorflow as tf
import numpy      as np
import math

from ConvNTMCell import ConvNTMCell


def ConvNTM_Network(encoder_input, decoder_input, encoder_channel, seq_channel, decoder_channel, training,
            filter_size=5, stride_size=1, padding="SAME"):
    """
    encoder_input: [time_step, batch_size, height, width, ch]
    """
    # CNN encoder
    Feature_enc, Enc_CNN1 = Encoder(encoder_input, encoder_channel) # [time_step, batch_size, height, width, ch]
    Feature_dec, Dec_CNN1 = Encoder(decoder_input, encoder_channel) # [time_step, batch_size, height, width, ch]
    # Sequence learning
    Output, Enc_seq, Enc_CNN2, Dec_seq, Dec_CNN2 = Sequence(input_enc=Feature_enc, input_dec=Feature_dec, channel_seq=seq_channel, 
                                                            channel_dec=decoder_channel, channel_enc=encoder_channel, training=training)
     # list of length depth with tensor [time_step*2, batch_size, height, width, ch]
    return Output, Enc_CNN1, Enc_seq, Enc_CNN2, Dec_CNN1, Dec_seq, Dec_CNN2

def Sequence(input_enc, input_dec, channel_seq, channel_dec, channel_enc, training, filter_size=5, stride_size=1, padding="SAME"):
    
    Time, Batch, height, width, ch = input_enc.get_shape().as_list()
    input_enc = tf.reshape(input_enc, shape=[Time, Batch, height*width*ch])
    
    layer = len(channel_seq)
    hidden = [0 for idx in channel_seq]
    channel_seq = [ch] + channel_seq
    Enc_seq = []
    Enc_CNN2 = [[] for idx in xrange(len(channel_dec))]
    
    pred = []
    for time_step in xrange(Time):
        for depth in xrange(layer):
            with tf.variable_scope("seq_layer"+str(depth)):
                try:
                    cell = ConvNTMCell(input_height=height, input_width=width, input_ch=channel_seq[depth], output_height=height, 
                                       output_width=width, output_ch=channel_seq[depth+1], 
                                       mem_height=height, mem_width=width, mem_size=4,
                                       filter_shape=[filter_size, filter_size, channel_seq[depth], channel_seq[depth+1]])
                    tf.get_variable_scope().reuse_variables()
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                if depth != 0:
                    if time_step != 0:
                        output, hidden[depth] = cell(output, state=hidden[depth])
                        Enc_seq.append(tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch]))
                    else:
                        output, hidden[depth] = cell(output)
                        Enc_seq.append(tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch]))
                else:
                    if time_step != 0:
                        output, hidden[depth] = cell(input_enc[time_step, :, :], state=hidden[depth])
                        Enc_seq.append(tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch]))
                    else:
                        output, hidden[depth] = cell(input_enc[time_step, :, :])
                        Enc_seq.append(tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch]))
        # output: [batch, height*width*ch]
        output = tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch])
        output, tmp = Decoder(output, channel_dec)
        for idx in xrange(len(tmp)):
            Enc_CNN2[idx].append(tmp[idx])
        pred.append(output)
        
    Enc_seq = tf.pack(Enc_seq, axis=0)
    for idx in xrange(len(Enc_CNN2)):
        Enc_CNN2[idx] = tf.pack(Enc_CNN2[idx], axis=0)
    
    Dec_CNN2 = [[] for idx in xrange(len(channel_dec))]
    Dec_seq = []
    for time_step in xrange(Time):
        for depth in xrange(layer):
            with tf.variable_scope("seq_layer"+str(depth)):
                try:
                    cell = ConvNTMCell(input_height=height, input_width=width, input_ch=channel_seq[depth], output_height=height, 
                                       output_width=width, output_ch=channel_seq[depth+1], 
                                       mem_height=height, mem_width=width, mem_size=4,
                                       filter_shape=[filter_size, filter_size, channel_seq[depth], channel_seq[depth+1]])
                    tf.get_variable_scope().reuse_variables()
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                if depth != 0:
                    output, hidden[depth] = cell(output, state=hidden[depth])
                    Dec_seq.append(tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch]))
                else:
                    if time_step != 0:
                        if training == False:
                            output, hidden[depth] = cell(feature, state=hidden[depth])
                            Dec_seq.append(tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch]))
                        else:
                            output, hidden[depth] = cell(input_dec[time_step-1, :, :], state=hidden[depth])
                            Dec_seq.append(tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch]))
                    else:
                        output, hidden[depth] = cell(input_enc[-1, :, :], state=hidden[depth])
                        Dec_seq.append(tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch]))
        # output: [batch, height*width*ch]
        output = tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch])
        output, tmp = Decoder(output, channel_dec)
        for idx in xrange(len(tmp)):
            Dec_CNN2[idx].append(tmp[idx])
        pred.append(output)
        
        # Encoder
        if time_step != 0:
            if training == False:
                feature = Encoder(pred[-1], channel_enc)
    Dec_seq = tf.pack(Dec_seq, axis=0)
    for idx in xrange(len(Enc_CNN2)):
        Dec_CNN2[idx] = tf.pack(Dec_CNN2[idx], axis=0)
    
    return tf.pack(pred), Enc_seq, Enc_CNN2, Dec_seq, Dec_CNN2 # [time_step*2, batch_size, height, width, ch]
        
def Decoder(input_, channel, filter_size=5, stride_size=1, padding="SAME"):
    """
    input_: [batch_size, height, width, ch]
    """
    Batch, height, width, ch = input_.get_shape().as_list()
    layer = len(channel)
    output = input_
    Feature = []
    for idx in xrange(layer):
        with tf.variable_scope("decoder_layer"+str(idx)):
            try:
                output = conv_layer(output, channel[idx], filter_size, stride_size, padding, tf.nn.elu)
                Feature.append(output)
            except ValueError:
                tf.get_variable_scope().reuse_variables()
                output = conv_layer(output, channel[idx], filter_size, stride_size, padding, tf.nn.elu)
                Feature.append(output)
    try:
        output = conv_layer(input_, out_ch=1, filter_size=1, stride_size=1, padding=padding, activation=None, scope="output_layer")
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        output = conv_layer(input_, out_ch=1, filter_size=1, stride_size=1, padding=padding, activation=None, scope="output_layer")
        
    return output, Feature # store feature maps in each layer [batch, height, width, ch]

def Encoder(input_, channel, filter_size=5, stride_size=1, padding="SAME"):
    """
    Encode input full time step and batch at the same time
    input_: [time_step, batch_size, height, width, ch]
    channel: number of channel in each layer
    """
    if len(input_.get_shape().as_list())==5:
        time_step, batch_size, height, width, ch = input_.get_shape().as_list()
        one_data = False
    else:
        batch_size, height, width, ch = input_.get_shape().as_list()
        one_data = True
    layer = len(channel)
    if layer == 0:
        return input_
    if one_data == False:
        output = tf.reshape(input_, shape=[-1, height, width, ch])
    else:
        output = input_
    
    Feature = []
    for idx in xrange(layer):
        with tf.variable_scope("encoder_layer"+str(idx)):
            try:
                output = conv_layer(output, channel[idx], filter_size, stride_size, padding, tf.nn.elu)
                Feature.append(output)
                if one_data == False:
                    Feature[-1] = tf.reshape(Feature[-1], shape=[time_step, batch_size, height, width, channel[-1]])
            except ValueError:
                tf.get_variable_scope().reuse_variables()
                output = conv_layer(output, channel[idx], filter_size, stride_size, padding, tf.nn.elu)
                Feature.append(output)
                if one_data == False:
                    Feature[-1] = tf.reshape(Feature[-1], shape=[time_step, batch_size, height, width, channel[-1]])
    if one_data == False:
        output = tf.reshape(output, shape=[time_step, batch_size, height, width, channel[-1]])
        
    # output: [time_step, batch_size, height, width, ch] or [batch_size, height, width, ch]
    # Feature: list of lenght layer size, where each element [time_step, batch_size, height, width, ch]
    return output, Feature

def Output_layer(input_, channel, filter_size=1, stride_size=1, padding="SAME"):
    """
    Final output layer with filter of shape 1 and without sigmoid
    input_: [batch_size, height, width, ch]
    channel: channel of output
    """
    try:
        output = conv_layer(input_, channel, filter_size, stride_size, padding, activation=None)
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        output = conv_layer(input_, channel, filter_size, stride_size, padding, activation=None)
    return output # [batch_size, height, width, ch]
    
def conv_layer(input_, out_ch, filter_size, stride_size, padding, activation=None, scope=None):
    """
    input_: [batch_size, height, width, ch]
    """
    with tf.variable_scope(scope or "conv_layer"):
        in_ch = input_.get_shape()[3]
        # initialization of weights
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)
        W = tf.get_variable(name="weights", shape=[filter_size, filter_size, in_ch, out_ch], initializer=initializer)
        b = tf.get_variable(name="biases", shape=[out_ch, ], initializer=tf.constant_initializer(0.1))
        # conv_layer
        output = tf.nn.conv2d(input_, filter=W, strides=[1, stride_size, stride_size, 1], padding=padding)
        output = tf.nn.bias_add(output, b)
        # activation function
        if activation is not None:
            return activation(output)
        else:
            return output
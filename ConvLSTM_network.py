import tensorflow as tf
import numpy      as np
import math

from ConvNTMCell import ConvLSTMCell


def ConvLSTM_Network(encoder_input, decoder_input, encoder_channel, seq_channel, decoder_channel, training,
            filter_size=5, stride_size=1, padding="SAME"):
    """
    encoder_input: [time_step, batch_size, height, width, ch]
    """
    # CNN encoder
    Feature_enc = Encoder(encoder_input, encoder_channel) # [time_step, batch_size, height, width, ch]
    Feature_dec = Encoder(decoder_input, encoder_channel) # [time_step, batch_size, height, width, ch]
    # Sequence learning
    Output = Sequence(input_enc=Feature_enc, input_dec=Feature_dec, channel_seq=seq_channel, channel_dec=decoder_channel, 
                      channel_enc=encoder_channel, training=training)
    return Output # list of length depth with tensor [time_step*2, batch_size, height, width, ch]

def Sequence(input_enc, input_dec, channel_seq, channel_dec, channel_enc, training, filter_size=5, stride_size=1, padding="SAME"):
    
    Time, Batch, height, width, ch = input_enc.get_shape().as_list()
    input_enc = tf.reshape(input_enc, shape=[Time, Batch, height*width*ch])
    
    layer = len(channel_seq)
    hidden = [tf.zeros(shape=[Batch, height*width*idx], dtype=tf.float32) for idx in channel_seq]
    channel_seq = [ch] + channel_seq
    pred = []
    for time_step in xrange(Time):
        for depth in xrange(layer):
            with tf.variable_scope("seq_layer"+str(depth)):
                try:
                    cell = ConvLSTMCell(input_height=height, input_width=width, input_ch=channel_seq[depth], output_height=height, 
                                        output_width=width, output_ch=channel_seq[depth+1], 
                                        filter_shape=[filter_size, filter_size, channel_seq[depth], channel_seq[depth+1]])
                    tf.get_variable_scope().reuse_variables()
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                
                if depth != 0:
                    output, hidden[depth] = cell(output, state=hidden[depth])
                else:
                    output, hidden[depth] = cell(input_enc[time_step, :, :], state=hidden[depth])
        # output: [batch, height*width*ch]
        output = tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch])
        output = Decoder(output, channel_dec)
        pred.append(output)
    
    for time_step in xrange(Time):
        for depth in xrange(layer):
            with tf.variable_scope("seq_layer"+str(depth)):
                try:
                    cell = ConvLSTMCell(input_height=height, input_width=width, input_ch=channel_seq[depth], output_height=height, 
                                        output_width=width, output_ch=channel_seq[depth+1], 
                                        filter_shape=[filter_size, filter_size, channel_seq[depth], channel_seq[depth+1]])
                    tf.get_variable_scope().reuse_variables()
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                
                if depth != 0:
                    output, hidden[depth] = cell(output, state=hidden[depth])
                else:
                    if time_step != 0:
                        if training == False:
                            output, hidden[depth] = cell(feature, state=hidden[depth])
                        else:
                            output, hidden[depth] = cell(input_dec[time_step-1, :, :], state=hidden[depth])
                    else:
                        output, hidden[depth] = cell(input_enc[-1, :, :], state=hidden[depth])
        # output: [batch, height*width*ch]
        output = tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch])
        output = Decoder(output, channel_dec)
        pred.append(output)
        
        # Encoder
        if time_step != 0:
            if training == False:
                feature = Encoder(pred[-1], channel_enc)
                
        
    return tf.pack(pred) # [time_step*2, batch_size, height, width, ch]
        
def Decoder(input_, channel, filter_size=5, stride_size=1, padding="SAME"):
    """
    input_: [batch_size, height, width, ch]
    """
    Batch, height, width, ch = input_.get_shape().as_list()
    layer = len(channel)
    output = input_
    for idx in xrange(layer):
        with tf.variable_scope("decoder_layer"+str(idx)):
            try:
                output = conv_layer(output, channel[idx], filter_size, stride_size, padding, tf.nn.elu)
            except ValueError:
                tf.get_variable_scope().reuse_variables()
                output = conv_layer(output, channel[idx], filter_size, stride_size, padding, tf.nn.elu)
    try:
        output = conv_layer(input_, out_ch=1, filter_size=1, stride_size=1, padding=padding, activation=None, scope="output_layer")
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        output = conv_layer(input_, out_ch=1, filter_size=1, stride_size=1, padding=padding, activation=None, scope="output_layer")
        
    return output

def Network(encoder_input, decoder_input, encoder_channel, seq_enc_channel, seq_dec_channel, decoder_channel, training,
            filter_size=5, stride_size=1, padding="SAME"):
    """
    Overall network
    encoder_input: [time_step, batch_size, height, width, ch]
    encoder_channel: a list specifying number of channel of CNN encoder at each layer
    seq_enc_channel: a list specifying number of channel of seq encoder at each layer
    """
    # CNN encoder
    Feature = Encoder(encoder_input, encoder_channel) # [time_step, batch_size, height, width, ch]
    # Seq encoder
    Encode_State = Seq_Enc(Feature, seq_enc_channel) # list of length depth with tensor [batch_size, state_size]
    # Seq decoder
    Pred = Seq_Dec(decoder_input, seq_dec_channel, decoder_channel, encoder_input[-1, :, :, :, :], Encode_State, training=training)
    
    return Pred # [time_step, batch_size, height, width, ch]
    
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
    for idx in xrange(layer):
        with tf.variable_scope("encoder_layer"+str(idx)):
            try:
                output = conv_layer(output, channel[idx], filter_size, stride_size, padding, tf.nn.elu)
            except ValueError:
                tf.get_variable_scope().reuse_variables()
                output = conv_layer(output, channel[idx], filter_size, stride_size, padding, tf.nn.elu)
    if one_data == False:
        output = tf.reshape(output, shape=[time_step, batch_size, height, width, channel[-1]])
    return output # [time_step, batch_size, height, width, ch] or [batch_size, height, width, ch]

def Seq_Enc(input_, channel, filter_size=5, stride_size=1, padding="SAME"):
    """
    input_: [time_step, batch_size, height, width, ch]
    channel: number of channel in each layer
    """
    Time, Batch, height, width, ch = input_.get_shape().as_list()
    input_ = tf.reshape(input_, shape=[Time, Batch, height*width*ch])
    
    layer = len(channel)
    hidden = [tf.zeros(shape=[Batch, height*width*ch], dtype=tf.float32)] * layer
    channel = [ch] + channel
    for time_step in xrange(Time):
        for depth in xrange(layer):
            with tf.variable_scope("seq_enc_layer"+str(depth)):
                try:
                    cell = ConvLSTMCell(input_height=height, input_width=width, input_ch=channel[depth], output_height=height, 
                                        output_width=width, output_ch=channel[depth+1], 
                                        filter_shape=[filter_size, filter_size, channel[depth], channel[depth+1]])
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                    cell = ConvLSTMCell(input_height=height, input_width=width, input_ch=channel[depth], output_height=height, 
                                        output_width=width, output_ch=channel[depth+1], 
                                        filter_shape=[filter_size, filter_size, channel[depth], channel[depth+1]])
                if depth != 0:
                    try:
                        output, hidden[depth] = cell(output, state=hidden[depth])
                    except ValueError:
                        tf.get_variable_scope().reuse_variables()
                        output, hidden[depth] = cell(output, state=hidden[depth])
                else:
                    try:
                        # output: [batch_size, state_size] at current time step
                        output, hidden[depth] = cell(input_[time_step, :, :], state=hidden[depth])
                    except ValueError:
                        tf.get_variable_scope().reuse_variables()
                        # output: [batch_size, state_size] at current time step
                        output, hidden[depth] = cell(input_[time_step, :, :], state=hidden[depth])
                        
    return hidden # list of length depth with tensor [batch_size, state_size]
    
def Seq_Dec(input_, channel_seq, channel_dec, first_input, hidden, filter_size=5, stride_size=1, padding="SAME", training=True):
    """
    input_: [time_step, batch_size, height, width, ch]
    channel: number of channel in each layer
    first_input: [batch_size, height, width, ch], the last frame of input sequence to feed into the sequence model at first time step
    hidden: previous encoded code, which is a list of length depth with tensor [batch_size, state_size]
    """
    Time, Batch, height, width, ch = input_.get_shape().as_list()
    pred = []
    
    # input_ of shape [time_step, batch_size, height, width, ch]
    layer = len(channel_seq)
    channel_seq = [ch] + channel_seq
    for time_step in xrange(Time):
        # Before seq2seq model, we first feed into Decoder
        if time_step != 0:
            if training == True:
                output = Decoder(input_[time_step-1, :, :, :], channel_dec)
            else:
                output = Decoder(tf.sigmoid(pred[-1]), channel_dec)
        else:
            output = Decoder(first_input, channel_dec)
        channel_seq[0] = output.get_shape().as_list()[3]
        # Start seq2seq
        # output: [batch_size, height, width, ch]
        for depth in xrange(layer):
            #with tf.variable_scope("seq_dec_layer"+str(depth)):
            with tf.variable_scope("seq_enc_layer"+str(depth)):
                try:
                    cell = ConvLSTMCell(input_height=height, input_width=width, input_ch=channel_seq[depth], output_height=height, 
                                        output_width=width, output_ch=channel_seq[depth+1], 
                                        filter_shape=[filter_size, filter_size, channel_seq[depth], channel_seq[depth+1]])
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                    cell = ConvLSTMCell(input_height=height, input_width=width, input_ch=channel_seq[depth], output_height=height, 
                                        output_width=width, output_ch=channel_seq[depth+1], 
                                        filter_shape=[filter_size, filter_size, channel_seq[depth], channel_seq[depth+1]])
                h, w, c = output.get_shape().as_list()[1:]
                try:
                    output, hidden[depth] = cell(tf.reshape(output, shape=[-1, h*w*c]), state=hidden[depth])
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                    output, hidden[depth] = cell(tf.reshape(output, shape=[-1, h*w*c]), state=hidden[depth])
        # output: [batch_size, h*w*c]
        output = tf.reshape(output, shape=[-1, cell.output_height, cell.output_width, cell.output_ch])
        output = Output_layer(output, first_input.get_shape()[3])
        pred.append(output) # list of length time_step with tensor [batch_size, height, width, ch]
        
    return tf.pack(pred)
"""    
def Decoder(input_, channel, filter_size=5, stride_size=1, padding="SAME"):
"""
"""
    #Decode the input time step by time step
    #input_: [batch_size, height, width, ch]
    #channel: number of channel in each layer
"""
"""
    _, height, width, ch = input_.get_shape().as_list()
    layer = len(channel)
    output = tf.reshape(input_, shape=[-1, height, width, ch])
    for idx in xrange(layer):
        with tf.variable_scope("encoder_layer"+str(idx)):
            try:
                output = conv_layer(output, channel[idx], filter_size, stride_size, padding, tf.nn.elu, scope="encoder_layer"+str(idx))
            except ValueError:
                tf.get_variable_scope().reuse_variables()
                output = conv_layer(output, channel[idx], filter_size, stride_size, padding, tf.nn.elu, scope="encoder_layer"+str(idx))
    return output # [batch_size, height, width, ch]
"""
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
            

    
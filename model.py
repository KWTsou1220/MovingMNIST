import tensorflow as tf
import numpy      as np
import math
from seq2seq_model import seq2seq

from ConvLSTM_network import *
from ConvNTM_network import *


class MovingMNISTConvLSTM(object):
    def __init__(self, input_height, input_width, input_ch, time_step, batch_size, encoder_channel, seq_enc_channel, seq_dec_channel,
                 decoder_channel, LR):
        # Basic setting
        self.input_height = input_height
        self.input_width  = input_width
        self.input_ch     = input_ch
        self.time_step    = time_step
        self.batch_size   = batch_size
        self.encoder_channel = encoder_channel
        #self.seq_enc_channel = seq_enc_channel
        #self.seq_dec_channel = seq_dec_channel
        self.seq_channel = seq_enc_channel
        self.decoder_channel = decoder_channel
        self.LR = LR
        
        # placeholder: [time_step, batch_size, h, w, ch]
        self.x = tf.placeholder(tf.float32, [time_step, batch_size, input_height, input_width, input_ch]) 
        self.y = tf.placeholder(tf.float32, [time_step, batch_size, input_height, input_width, input_ch])
        self.training = tf.placeholder(tf.bool)
        
        # Network template
        network_template = tf.make_template('network', ConvLSTM_Network)
        self.pred_logits = network_template(self.x, self.y, self.encoder_channel, self.seq_channel, self.decoder_channel, self.training)
        self.pred = tf.sigmoid(self.pred_logits) # [time_step*2, batch_size, height, width, ch]
        
        # Optimization
        self.compute_loss()
        self.optimizer = tf.train.AdamOptimizer(self.LR)
        grad_var       = self.optimizer.compute_gradients(self.loss)
        def GradientClip(grad):
            if grad is None:
                return grad
            #return tf.clip_by_norm(grad, 1)
            return tf.clip_by_value(grad, -1, 1)
        clip_grad_var = [(GradientClip(grad), var) for grad, var in grad_var ]
        self.train_op = self.optimizer.apply_gradients(clip_grad_var)
        
    def compute_loss(self):
        entropy_enc = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.reshape(self.pred_logits[0:9, :, :, :, :], [-1, self.input_height*self.input_width*self.input_ch]),
            targets=tf.reshape(self.x[1:, :, :, :, :], [-1, self.input_height*self.input_width*self.input_ch]),
            name="entropy_enc")
        entropy_dec = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.reshape(self.pred_logits[10:, :, :, :, :], [-1, self.input_height*self.input_width*self.input_ch]),
            targets=tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch]),
            name="entropy_enc")
        #entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=tf.reshape(self.pred_logits, [-1, self.input_height*self.input_width*self.input_ch]), 
        #    targets=tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch]), 
        #    name="entropy")
        #entropy = tf.pow(self.pred - tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch]), 2)
        
        #entropy = (tf.maximum(self.pred_logits, 0) - 
        #           tf.multiply(self.pred_logits, tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch])) + 
        #           tf.log(1+tf.exp(-tf.abs(self.pred_logits))))
        
        #target  = tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch])
        #entropy = -1*(tf.multiply(target, tf.log(self.pred+1e-6)) + tf.multiply(1-target, tf.log(1-self.pred+1e-6)))
        
        #self.loss = tf.reduce_mean(tf.reduce_sum(entropy, axis=1))
        self.pred_loss = tf.reduce_mean(tf.reduce_sum(entropy_dec, axis=1))
        self.loss = tf.reduce_mean(tf.reduce_sum(entropy_enc, axis=1)) + self.pred_loss
        
class MovingMNISTConvNTM(object):
    def __init__(self, input_height, input_width, input_ch, time_step, batch_size, encoder_channel, seq_enc_channel, seq_dec_channel,
                 decoder_channel, LR):
        # Basic setting
        self.input_height = input_height
        self.input_width  = input_width
        self.input_ch     = input_ch
        self.time_step    = time_step
        self.batch_size   = batch_size
        self.encoder_channel = encoder_channel
        #self.seq_enc_channel = seq_enc_channel
        #self.seq_dec_channel = seq_dec_channel
        self.seq_channel = seq_enc_channel
        self.decoder_channel = decoder_channel
        self.LR = LR
        
        # placeholder: [time_step, batch_size, h, w, ch]
        self.x = tf.placeholder(tf.float32, [time_step, batch_size, input_height, input_width, input_ch]) 
        self.y = tf.placeholder(tf.float32, [time_step, batch_size, input_height, input_width, input_ch])
        self.training = tf.placeholder(tf.bool)
        
        # Network template
        network_template = tf.make_template('network', ConvNTM_Network)
        self.pred_logits, self.Enc_CNN1, self.Enc_seq, self.Enc_CNN2, self.Dec_CNN1, self.Dec_seq, self.Dec_CNN2 = network_template(
            self.x, self.y, self.encoder_channel, self.seq_channel, self.decoder_channel, self.training)
        self.pred = tf.sigmoid(self.pred_logits) # [time_step*2, batch_size, height, width, ch]
        
        # Optimization
        self.compute_loss()
        self.optimizer = tf.train.AdamOptimizer(self.LR)
        grad_var       = self.optimizer.compute_gradients(self.loss)
        def GradientClip(grad):
            if grad is None:
                return grad
            #return tf.clip_by_norm(grad, 1)
            return tf.clip_by_value(grad, -1, 1)
        clip_grad_var = [(GradientClip(grad), var) for grad, var in grad_var ]
        self.train_op = self.optimizer.apply_gradients(clip_grad_var)
        
    def compute_loss(self):
        entropy_enc = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.reshape(self.pred_logits[0:9, :, :, :, :], [-1, self.input_height*self.input_width*self.input_ch]),
            targets=tf.reshape(self.x[1:, :, :, :, :], [-1, self.input_height*self.input_width*self.input_ch]),
            name="entropy_enc")
        entropy_dec = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.reshape(self.pred_logits[10:, :, :, :, :], [-1, self.input_height*self.input_width*self.input_ch]),
            targets=tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch]),
            name="entropy_enc")
        #entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=tf.reshape(self.pred_logits, [-1, self.input_height*self.input_width*self.input_ch]), 
        #    targets=tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch]), 
        #    name="entropy")
        #entropy = tf.pow(self.pred - tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch]), 2)
        
        #entropy = (tf.maximum(self.pred_logits, 0) - 
        #           tf.multiply(self.pred_logits, tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch])) + 
        #           tf.log(1+tf.exp(-tf.abs(self.pred_logits))))
        
        #target  = tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch])
        #entropy = -1*(tf.multiply(target, tf.log(self.pred+1e-6)) + tf.multiply(1-target, tf.log(1-self.pred+1e-6)))
        
        #self.loss = tf.reduce_mean(tf.reduce_sum(entropy, axis=1))
        self.pred_loss = tf.reduce_mean(tf.reduce_sum(entropy_dec, axis=1))
        self.loss = tf.reduce_mean(tf.reduce_sum(entropy_enc, axis=1)) + self.pred_loss
        
        
class MovingMNISTModel(object):
    def __init__(self, encode_arch, decode_arch, input_height, input_width, input_ch, output_height, output_width, output_ch, batch_size, 
                 LR, time_step=10, tensor=False):
        # input size
        self.input_height = input_height
        self.input_width  = input_width
        self.input_ch     = input_ch
        # output size
        self.output_height = output_height
        self.output_width  = output_width
        self.output_ch     = output_ch
        # tensor or not
        self.tensor = tensor
        # other settings
        self.batch_size = batch_size
        self.time_step  = time_step
        self.LR = LR
        self.encode_arch = encode_arch
        self.decode_arch = decode_arch
        self.num_layer_enc = len(encode_arch)
        self.num_layer_dec = len(decode_arch)
        
        # placeholder
        self.x = tf.placeholder(tf.float32, [None, input_height, input_width, input_ch]) # [time_step x batch_size, h, w, ch]
        self.y = tf.placeholder(tf.float32, [None, output_height, output_width, output_ch])
        self.feed_previous = tf.placeholder(tf.bool)
        
        # feed forward
        with tf.variable_scope('Model'):
            """
            with tf.variable_scope('Encoder'):
                self.Encoder()
            with tf.variable_scope('Decoder'):
                self.Decoder()
            """
            self.enc_cell, self.dec_cell, init_state = self.construct()
            
            
            # Output
            encoder_input = tf.reshape(self.x, shape=[self.time_step, self.batch_size, self.input_height*self.input_width*self.input_ch])
            decoder_input = tf.reshape(self.y, shape=[self.time_step, self.batch_size, self.input_height*self.input_width*self.input_ch])
            # [time_step x batch_size, out_h x out_w x out_ch]
            self.pred_logits = seq2seq(self.enc_cell, self.dec_cell, encoder_input, decoder_input, init_state, self.tensor, 
                                       feed_previous=self.feed_previous)
            #self.pred_logits = self.one_layer_seq2seq(self.enc_cell, self.dec_cell, encoder_input, decoder_input, init_state, 
            #                                         self.feed_previous)
            self.pred_logits = tf.reshape(self.pred_logits, shape=[-1, self.input_height*self.input_width*self.input_ch])
            self.pred = tf.sigmoid(self.pred_logits)
        
        # optimization
        self.compute_loss()
        self.optimizer = tf.train.AdamOptimizer(self.LR)
        grad_var       = self.optimizer.compute_gradients(self.loss)
        def GradientClip(grad):
            if grad is None:
                return grad
            #return tf.clip_by_norm(grad, 1)
            return tf.clip_by_value(grad, -1, 1)
        clip_grad_var = [(GradientClip(grad), var) for grad, var in grad_var ]
        self.train_op = self.optimizer.apply_gradients(clip_grad_var)

    def compute_loss(self):
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.pred_logits, 
            targets=tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch]), 
            name="entropy")
        #entropy = tf.pow(self.pred - tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch]), 2)
        
        #entropy = (tf.maximum(self.pred_logits, 0) - 
        #           tf.multiply(self.pred_logits, tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch])) + 
        #           tf.log(1+tf.exp(-tf.abs(self.pred_logits))))
        
        #target  = tf.reshape(self.y, [-1, self.input_height*self.input_width*self.input_ch])
        #entropy = -1*(tf.multiply(target, tf.log(self.pred+1e-6)) + tf.multiply(1-target, tf.log(1-self.pred+1e-6)))
        
        self.loss = tf.reduce_mean(tf.reduce_sum(entropy, axis=1))

    def construct(self):
        enc_cell = []
        dec_cell = []
        init_state = []
        # Encoder
        with tf.variable_scope("Encoder"):
            for idx in range(1, self.num_layer_enc):
                with tf.variable_scope("layer"+str(idx-1)):
                    if self.encode_arch['l'+str(idx)]['type'] == 'convNTM':
                        curr_shape = self.encode_arch['l'+str(idx-1)]['shape'] # [batch_size, height, width, ch]
                        next_shape = self.encode_arch['l'+str(idx)]['shape'] # [batch_size, height, width, ch]
                        cell = ConvNTMCell(input_height=curr_shape[1], input_width=curr_shape[2], input_ch=curr_shape[3], 
                                               output_height=next_shape[1], output_width=next_shape[2], output_ch=next_shape[3], 
                                               mem_height=next_shape[1], mem_width=next_shape[2], 
                                               mem_size=self.encode_arch['l'+str(idx)]['mem_size'],
                                               filter_shape=self.encode_arch['l'+str(idx)]['filter'], 
                                               stride_shape=self.encode_arch['l'+str(idx)]['stride'])
                        init_state_ = cell.init_state(self.batch_size, dtype=tf.float32)
                        enc_cell.append(cell)
                        init_state.append(init_state_)
                    elif self.encode_arch['l'+str(idx)]['type'] == 'convLSTM':
                        curr_shape = self.encode_arch['l'+str(idx-1)]['shape'] # [batch_size, height, width, ch]
                        next_shape = self.encode_arch['l'+str(idx)]['shape'] # [batch_size, height, width, ch]
                        cell = ConvLSTMCell(input_height=curr_shape[1], input_width=curr_shape[2], input_ch=curr_shape[3], 
                                                output_height=next_shape[1], output_width=next_shape[2], output_ch=next_shape[3], 
                                                filter_shape=self.encode_arch['l'+str(idx)]['filter'], 
                                                stride_shape=self.encode_arch['l'+str(idx)]['stride'])
                        init_state_ = cell.zero_state(self.batch_size, dtype=tf.float32)
                        enc_cell.append(cell)
                        init_state.append(init_state_)
                    elif self.encode_arch['l'+str(idx)]['type'] == 'LSTM':
                        curr_shape = self.encode_arch['l'+str(idx-1)]['shape'][0] # [batch_size, units]
                        next_shape = self.encode_arch['l'+str(idx)]['shape'][0] # [batch_size, units]
                        cell = tf.nn.rnn_cell.LSTMCell(num_units=next_shape, use_peepholes=False, forget_bias=1.0, state_is_tuple=False)
                        init_state_ = cell.zero_state(self.batch_size, dtype=tf.float32)
                        # init
                        cell(tf.zeros([self.batch_size, curr_shape]), init_state_, scope="LSTMCell")
                        enc_cell.append(cell)
                        init_state.append(init_state_)
                        
        # Decoder
        with tf.variable_scope("Decoder"):
            for idx in range(1, self.num_layer_dec):
                with tf.variable_scope("layer"+str(idx-1)):
                    if self.decode_arch['l'+str(idx)]['type'] == 'convNTM':
                        curr_shape = self.decode_arch['l'+str(idx-1)]['shape'] # [batch_size, height, width, ch]
                        next_shape = self.decode_arch['l'+str(idx)]['shape'] # [batch_size, height, width, ch]
                        cell = ConvNTMCell(input_height=curr_shape[1], input_width=curr_shape[2], input_ch=curr_shape[3], 
                                               output_height=next_shape[1], output_width=next_shape[2], output_ch=next_shape[3], 
                                               mem_height=next_shape[1], mem_width=next_shape[2], 
                                               mem_size=self.decode_arch['l'+str(idx)]['mem_size'],
                                               filter_shape=self.decode_arch['l'+str(idx)]['filter'], 
                                               stride_shape=self.decode_arch['l'+str(idx)]['stride'])
                        # init
                        #dec_cell(tf.zeros([self.batch_size, curr_shape[1]*curr_shape[2]*curr_shape[3]]), init_state)
                        #initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
                        dec_cell.append(cell)
                        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)
                        if idx == self.num_layer_dec-2:
                            W = self._weight_variable(self.decode_arch['l'+str(idx+1)]['filter'], initializer=initializer,
                                                      name="final_weights")
                            b = self._bias_variable([self.decode_arch['l'+str(idx+1)]['filter'][3],], name="final_biases")
                    elif self.decode_arch['l'+str(idx)]['type'] == 'convLSTM':
                        curr_shape = self.decode_arch['l'+str(idx-1)]['shape'] # [batch_size, height, width, ch]
                        next_shape = self.decode_arch['l'+str(idx)]['shape'] # [batch_size, height, width, ch]
                        cell = ConvLSTMCell(input_height=curr_shape[1], input_width=curr_shape[2], input_ch=curr_shape[3], 
                                                output_height=next_shape[1], output_width=next_shape[2], output_ch=next_shape[3], 
                                                filter_shape=self.decode_arch['l'+str(idx)]['filter'], 
                                                stride_shape=self.decode_arch['l'+str(idx)]['stride'])
                        # init                
                        #dec_cell(tf.zeros([self.batch_size, curr_shape[1]*curr_shape[2]*curr_shape[3]]), init_state)
                        #initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
                        dec_cell.append(cell)
                        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)
                        if idx == self.num_layer_dec-2:
                            W = self._weight_variable(self.decode_arch['l'+str(idx+1)]['filter'], initializer=initializer,
                                                      name="final_weights")
                            b = self._bias_variable([self.decode_arch['l'+str(idx+1)]['filter'][3],], name="final_biases")
                    elif self.decode_arch['l'+str(idx)]['type'] == 'LSTM':
                        curr_shape = self.decode_arch['l'+str(idx-1)]['shape'][0] # [batch_size, units]
                        next_shape = self.decode_arch['l'+str(idx)]['shape'][0] # [batch_size, units]
                        cell = tf.nn.rnn_cell.LSTMCell(num_units=next_shape, use_peepholes=False, forget_bias=1.0, state_is_tuple=False)
                        
                        # init
                        init_state_ = cell.zero_state(self.batch_size, dtype=tf.float32)
                        cell(tf.zeros([self.batch_size, curr_shape]), init_state_, scope="LSTMCell")
                        dec_cell.append(cell)
                        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)
                        if idx == self.num_layer_dec-2:
                            W = self._weight_variable([next_shape, self.input_height*self.input_width*self.output_ch], 
                                                      initializer=initializer, name="final_weights")
                            b = self._bias_variable([self.input_height*self.input_width*self.output_ch,], name="final_biases")
        return enc_cell, dec_cell, init_state
                    
        
        """    
        # Output
        encoder_input = tf.reshape(self.x, shape=[self.time_step, self.batch_size, self.input_height*self.input_width*self.input_ch])
        decoder_input = tf.reshape(self.y, shape=[self.time_step, self.batch_size, self.input_height*self.input_width*self.input_ch])
        # [time_step x batch_size, out_h x out_w x out_ch]
        self.pred_logits = self.one_layer_seq2seq(enc_cell, dec_cell, encoder_input, decoder_input, init_state, self.feed_previous)
        self.pred = tf.sigmoid(self.pred_logits)
        """                
        
    def one_layer_seq2seq(self, encoder_cell, decoder_cell, encoder_input, decoder_input, init_state, feed_previous=False, scope=None):
        """
        cell: tf.nn.rnn_cell object
        encoder_inputs: [time_steps, batch_size, input_size]
        decoder_inputs: [time_steps, batch_size, input_size]
        init_state: [batch_size, state_size]
        feed_previous: use preivous output or not
        """
        
        self.state  = init_state
        
        # Encode
        with tf.variable_scope("Encoder", reuse=True):
            for time_step in xrange(self.time_step):
                _, self.state = encoder_cell(encoder_input[time_step, :, :], self.state)
                
        # Decoder
        output = [] # without sigmoid!!!!
        layer = len(self.decode_arch)-1
        with tf.variable_scope("Decoder", reuse=True):
            for time_step in xrange(self.time_step):
                if time_step == 0:
                    #self.hidden, self.state = decoder_cell(tf.sigmoid(output[time_step]), self.state)
                    self.hidden, self.state = decoder_cell(encoder_input[-1, :, :], self.state)
                else:
                    if feed_previous == False:
                        self.hidden, self.state = decoder_cell(decoder_input[time_step-1, :, :], self.state)
                    else:
                        self.hidden, self.state = decoder_cell(output[time_step-1], self.state)
                
                # predict
                if len(self.decode_arch['l'+str(layer)]['shape'])>1:
                    tmp = tf.reshape(self.hidden, shape=self.decode_arch['l'+str(layer-1)]['shape'])
                    W = tf.get_variable("final_weights")
                    b = tf.get_variable("final_biases")
                    tmp = tf.nn.conv2d(tmp, filter=W, strides=self.decode_arch['l'+str(layer)]['stride'], padding="SAME")
                    tmp = tf.nn.bias_add(tmp, b)
                    tmp = tf.reshape(tmp, self.decode_arch['l'+str(layer)]['shape'])
                    output.append(tmp)
                else:
                    W = tf.get_variable("final_weights")
                    b = tf.get_variable("final_biases")
                    tmp = tf.nn.bias_add(tf.matmul(self.hidden, W), b)
                    output.append(tmp) # [time_step x batch_size, input_size]
        
        return tf.reshape(tf.pack(output), shape=[-1, self.output_height*self.output_width*self.output_ch]) 
        
        
    def Encoder(self):
        # [time_step x batch_size, in_height x in_width x in_ch]
        self.Neurons_enc = {'l0':tf.reshape(self.x, [-1, self.input_height*self.input_width*self.input_ch])} 
        self.States_enc  = {}
        self.init_state_enc = {}
        for idx in range(1, self.num_layer_enc):
            if self.encode_arch['l'+str(idx)]['type'] == 'convNTM':
                curr_shape = self.encode_arch['l'+str(idx-1)]['shape'] # [batch_size, height, width, ch]
                next_shape = self.encode_arch['l'+str(idx)]['shape'] # [batch_size, height, width, ch]
                convNtm_cell = ConvNTMCell(input_height=curr_shape[1], input_width=curr_shape[2], input_ch=curr_shape[3], 
                                           output_height=next_shape[1], output_width=next_shape[2], output_ch=next_shape[3], 
                                           mem_height=next_shape[1], mem_width=next_shape[2], 
                                           mem_size=self.encode_arch['l'+str(idx)]['mem_size'],
                                           filter_shape=self.encode_arch['l'+str(idx)]['filter'], 
                                           stride_shape=self.encode_arch['l'+str(idx)]['stride'])
                self.init_state_enc.update({'l'+str(idx):convNtm_cell.init_state(self.batch_size, dtype=tf.float32)})
                with tf.variable_scope('l'+str(idx)):
                    neurons, final_state = tf.nn.dynamic_rnn(
                        convNtm_cell, tf.reshape(self.Neurons_enc['l'+str(idx-1)], 
                                                 [self.time_step, self.batch_size, curr_shape[1]*curr_shape[2]*curr_shape[3]]),
                        initial_state=self.init_state_enc['l'+str(idx)], time_major=True)
                neurons = tf.reshape(neurons, [-1, next_shape[1]*next_shape[1]*next_shape[3]])
                self.Neurons_enc.update({'l'+str(idx):neurons}) # [time_step x batch_size, height x width x ch]
                self.States_enc.update({'l'+str(idx):final_state}) # [batch_size, height x width x ch]
            elif self.encode_arch['l'+str(idx)]['type'] == 'convLSTM':
                curr_shape = self.encode_arch['l'+str(idx-1)]['shape'] # [batch_size, height, width, ch]
                next_shape = self.encode_arch['l'+str(idx)]['shape'] # [batch_size, height, width, ch]
                convLSTM_cell = ConvLSTMCell(input_height=curr_shape[1], input_width=curr_shape[2], input_ch=curr_shape[3], 
                                             output_height=next_shape[1], output_width=next_shape[2], output_ch=next_shape[3], 
                                             filter_shape=self.encode_arch['l'+str(idx)]['filter'], 
                                             stride_shape=self.encode_arch['l'+str(idx)]['stride'])
                self.init_state_enc.update({'l'+str(idx):convLSTM_cell.zero_state(self.batch_size, dtype=tf.float32)})
                with tf.variable_scope('l'+str(idx)):
                    neurons, final_state = tf.nn.dynamic_rnn(
                        convLSTM_cell, tf.reshape(self.Neurons_enc['l'+str(idx-1)], 
                                                  [self.time_step, self.batch_size, curr_shape[1]*curr_shape[2]*curr_shape[3]]),
                        initial_state=self.init_state_enc['l'+str(idx)], time_major=True)
                neurons = tf.reshape(neurons, [-1, next_shape[1]*next_shape[1]*next_shape[3]])
                self.Neurons_enc.update({'l'+str(idx):neurons}) # [time_step x batch_size, height x width x ch]
                self.States_enc.update({'l'+str(idx):final_state}) # [batch_size, height x width x ch]
            elif self.encode_arch['l'+str(idx)]['type'] == 'output':
                pass
    def Decoder(self):
        # [time_step x batch_size, in_height x in_width x in_ch]
        self.Neurons_dec = {'l0':tf.zeros(shape=[self.time_step*self.batch_size, self.input_height*self.input_width*self.input_ch])}
        self.States_dec = {}
        self.init_state_dec = self.States_enc
        for idx in range(1, self.num_layer_dec):
            if self.decode_arch['l'+str(idx)]['type'] == 'convNTM':
                curr_shape = self.decode_arch['l'+str(idx-1)]['shape'] # [batch_size, height, width, ch]
                next_shape = self.decode_arch['l'+str(idx)]['shape'] # [batch_size, height, width, ch]
                convNtm_cell = ConvNTMCell(input_height=curr_shape[1], input_width=curr_shape[2], input_ch=curr_shape[3], 
                                           output_height=next_shape[1], output_width=next_shape[2], output_ch=next_shape[3], 
                                           mem_height=next_shape[1], mem_width=next_shape[2], 
                                           mem_size=self.decode_arch['l'+str(idx)]['mem_size'],
                                           filter_shape=self.decode_arch['l'+str(idx)]['filter'], 
                                           stride_shape=self.decode_arch['l'+str(idx)]['stride'])
                with tf.variable_scope('l'+str(idx)):
                    neurons, final_state = tf.nn.dynamic_rnn(
                        convNtm_cell, tf.reshape(self.Neurons_dec['l'+str(idx-1)], 
                                                 [self.time_step, -1, curr_shape[1]*curr_shape[2]*curr_shape[3]]),
                        initial_state=self.init_state_dec['l'+str(idx)], time_major=True)
                neurons = tf.reshape(neurons, [-1, next_shape[1]*next_shape[1]*next_shape[3]])
                self.Neurons_dec.update({'l'+str(idx):neurons}) # [time_step x batch_size, height x width x ch]
                self.States_dec.update({'l'+str(idx):final_state})
            elif self.decode_arch['l'+str(idx)]['type'] == 'convLSTM':
                curr_shape = self.decode_arch['l'+str(idx-1)]['shape'] # [batch_size, height, width, ch]
                next_shape = self.decode_arch['l'+str(idx)]['shape'] # [batch_size, height, width, ch]
                convLSTM_cell = ConvLSTMCell(input_height=curr_shape[1], input_width=curr_shape[2], input_ch=curr_shape[3], 
                                             output_height=next_shape[1], output_width=next_shape[2], output_ch=next_shape[3], 
                                             filter_shape=self.decode_arch['l'+str(idx)]['filter'], 
                                             stride_shape=self.decode_arch['l'+str(idx)]['stride'])
                with tf.variable_scope('l'+str(idx)):
                    neurons, final_state = tf.nn.dynamic_rnn(
                        convLSTM_cell, tf.reshape(self.Neurons_dec['l'+str(idx-1)], 
                                                  [self.time_step, -1, curr_shape[1]*curr_shape[2]*curr_shape[3]]),
                        initial_state=self.init_state_dec['l'+str(idx)], time_major=True)
                neurons = tf.reshape(neurons, [-1, next_shape[1]*next_shape[1]*next_shape[3]])
                self.Neurons_dec.update({'l'+str(idx):neurons}) # [time_step x batch_size, height x width x ch]
                self.States_dec.update({'l'+str(idx):final_state})
            elif self.decode_arch['l'+str(idx)]['type'] == 'output':
                curr_shape = self.decode_arch['l'+str(idx-1)]['shape'] # [batch_size, height, width, ch]
                next_shape = self.decode_arch['l'+str(idx)]['shape'] # [batch_size, height, width, ch]
                with tf.variable_scope('l'+str(idx)):
                    neurons = self.conv_layer(
                        tf.reshape(self.Neurons_dec['l'+str(idx-1)], [-1, curr_shape[1], curr_shape[2], curr_shape[3]]),
                        filter_shape=self.decode_arch['l'+str(idx)]['filter'], 
                        stride_shape=self.decode_arch['l'+str(idx)]['stride'],
                        activation=None)
                    neurons = tf.reshape(neurons, [-1, next_shape[1]*next_shape[1]*next_shape[3]])
                self.Neurons_dec.update({'l'+str(idx):neurons}) # [batch_size x time_step, height*width*ch]
                self.pred_logits = self.Neurons_dec['l'+str(idx)]
                #self.pred        = tf.sigmoid(self.pred_logits) # [batch_size x time_step, height*width*ch]
                self.pred        = tf.nn.relu(self.pred_logits)
                
    def conv_layer(self, x, filter_shape, stride_shape, padding='SAME', activation=tf.nn.relu):
        """
        with non linear function
        x: (batch_size x time_step) x height x width x ch
        """
        
        # initialization
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        W = self._weight_variable(filter_shape, initializer=initializer)
        b = self._bias_variable([filter_shape[3],])
        
        # convolution
        y = tf.nn.conv2d(x, filter=W, strides=stride_shape, padding=padding)
        y = tf.nn.bias_add(y, b)
        if activation is not None:
            return activation(y) # [batch_size x time_step, height, width, ch]
        else:
            return y
        
    def _weight_variable(self, shape, name='weights', initializer=tf.random_normal_initializer(mean=0., stddev=0.001,)):
        #d = 4*math.sqrt(6/(shape[0]+shape[1]))
        #initializer = tf.random_uniform_initializer(minval=-d, maxval=d, )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)



    
    
    
    
    
    
    
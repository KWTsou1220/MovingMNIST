from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce

import numpy      as np
import tensorflow as tf

from tensorflow.python.ops import array_ops

from utils import *
from ops   import *

class ConvNTMCell(tf.nn.rnn_cell.RNNCell):
    def init_state(self, batch_size, dtype=tf.float32):
        zero_dic = {}
        zero_dic.update({'M'      :tf.random_normal([batch_size, self.mem_width*self.mem_height*self.mem_size*self.output_ch], 
                                                    mean=0.0, stddev=0.001, dtype=dtype)})
        zero_dic.update({'read_w' :tf.zeros([batch_size, self.mem_size*self.output_ch], dtype=dtype)})
        zero_dic.update({'write_w':tf.zeros([batch_size, self.mem_size*self.output_ch], dtype=dtype)})
        zero_dic.update({'read'   :tf.zeros([batch_size, self.mem_height*self.mem_width*self.output_ch], dtype=dtype)})
        zero_dic.update({'output' :tf.zeros([batch_size, self.output_height*self.output_width*self.output_ch], dtype=dtype)})
        zero_dic.update({'hidden' :tf.zeros([batch_size, self.output_height*self.output_width*self.output_ch], dtype=dtype)})
        self.batch_size = batch_size
        zero = self.state_dic_to_state(zero_dic)
        return zero
        
    def __init__(self, input_height, input_width, input_ch, output_height, output_width, output_ch, mem_height, mem_width, mem_size,
                 filter_shape, stride_shape=[1, 1, 1, 1], padding="SAME", read_head_size=1, write_head_size=1):
        # initialize configs
        # input size
        self.input_height = input_height
        self.input_width  = input_width
        self.input_ch     = input_ch
        # output size
        self.output_height = output_height
        self.output_width  = output_width
        self.output_ch     = output_ch
        # memory size
        self.mem_height = output_height # mem_height # mem_height = output_height
        self.mem_width  = output_width # mem_width = output_width
        self.mem_size   = mem_size
        # number of read write head
        self.read_head_size  = read_head_size
        self.write_head_size = write_head_size
        # filter, stride, padding
        self.filter_shape = filter_shape
        self.stride_shape = stride_shape
        self.padding = padding
        
        # initialization:
        self.__call__(tf.zeros([2, self.input_height*self.input_width*self.input_ch]), self.init_state(2, dtype=tf.float32))
    
    @property
    def input_size(self):
        return self.input_height*self.input_width*self.input_ch

    @property
    def output_size(self):
        return self.output_height*self.output_width*self.output_ch

    @property
    def state_size(self):
        return (self.mem_height*self.mem_width*self.mem_size*self.output_ch + self.mem_size*self.output_ch*2 + 
                self.mem_height*self.mem_width*self.output_ch + self.output_height*self.output_width*self.output_ch*2)
    
    def __call__(self, input_, state=None, scope=None):
        """
        input_: of shape [batch_size, state_size=(input_height x input_width x input_ch)]
        output: of shape [batch_size, state_size=(output_height x output_width x output_ch)]
        """
        self.batch_size = input_.get_shape().as_list()[0]
        input_ = self.reshape(input_, "input")
        if state == None:
            state = self.init_state(self.batch_size)
        state_dic = self.state_to_state_dic(state)
            
        M_prev = state_dic['M']
        read_w_prev = state_dic['read_w']
        write_w_prev = state_dic['write_w']
        read_prev = state_dic['read'] # [batch_size, mem_height, mem_width, output_ch]
        output_prev = state_dic['output']
        hidden_prev = state_dic['hidden']
        
        # build a controller
        output, hidden = self.build_controller(input_, read_prev, output_prev, hidden_prev)
        # build a memory
        M, read_w, write_w, read = self.build_memory(M_prev, read_w_prev, write_w_prev, output)
        
        state_dic = {
            'M'      : M,
            'read_w' : read_w,
            'write_w': write_w,
            'read'   : read,
            'output' : output,
            'hidden' : hidden,
        }
        state = self.state_dic_to_state(state_dic)
        
        return tf.reshape(state_dic['output'], shape=[self.batch_size, self.output_height*self.output_width*self.output_ch]), state
    
    # ============================================== State To Dictionary ==============================================
    def state_dic_to_state(self, state_dic):
        M       = tf.reshape(state_dic['M'], shape=[self.batch_size, self.mem_height*self.mem_width*self.mem_size*self.output_ch])
        read_w  = tf.reshape(state_dic['read_w'], shape=[self.batch_size, self.mem_size*self.output_ch])
        write_w = tf.reshape(state_dic['write_w'], shape=[self.batch_size, self.mem_size*self.output_ch])
        read    = tf.reshape(state_dic['read'], shape=[self.batch_size, self.mem_height*self.mem_width*self.output_ch])
        output  = tf.reshape(state_dic['output'], shape=[self.batch_size, self.output_height*self.output_width*self.output_ch])
        hidden  = tf.reshape(state_dic['hidden'], shape=[self.batch_size, self.output_height*self.output_width*self.output_ch])

        state = tf.concat(1, [M, read_w, write_w, read, output, hidden])
        return state
    def state_to_state_dic(self, state):
        start_idx = 0
        M = tf.slice(state, [0, start_idx], [-1, self.mem_height*self.mem_width*self.mem_size*self.output_ch])
        start_idx += self.mem_height*self.mem_width*self.mem_size*self.output_ch
        read_w = tf.slice(state, [0, start_idx], [-1, self.mem_size*self.output_ch])
        start_idx += self.mem_size*self.output_ch
        write_w = tf.slice(state, [0, start_idx], [-1, self.mem_size*self.output_ch])
        start_idx += self.mem_size*self.output_ch
        read = tf.slice(state, [0, start_idx], [-1, self.output_height*self.output_width*self.output_ch])
        start_idx += self.output_height*self.output_width*self.output_ch
        output = tf.slice(state, [0, start_idx], [-1, self.output_height*self.output_width*self.output_ch])
        start_idx += self.output_height*self.output_width*self.output_ch
        hidden = tf.slice(state, [0, start_idx], [-1, self.output_height*self.output_width*self.output_ch])
        
        M = self.reshape(M, reshape_type="memory")
        read_w  = self.reshape(read_w, reshape_type="head")
        write_w = self.reshape(write_w, reshape_type="head")
        read    = self.reshape(read, reshape_type="controller")
        output  = self.reshape(output, reshape_type="controller")
        hidden  = self.reshape(hidden, reshape_type="controller")
        
        state_dic = {
            'M'      : M,
            'read_w' : read_w,
            'write_w': write_w,
            'read'   : read,
            'output' : output,
            'hidden' : hidden,
        }
        return state_dic
    # =================================================================================================================
    
    # =============================================== Build Controller ===============================================
    # Define the operation in the controller: comput current output and hidden based on previous state and input
    def build_controller(self, input_, read_prev, output_prev, hidden_prev):
        """
        input_, output_prev, hidden_prev: have been reshaped
        """
        with tf.variable_scope("controller"):
            def new_gate(gate_name):
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
                W = tf.get_variable(shape=[self.mem_height, self.mem_width, self.output_ch], 
                                    initializer=initializer, name='weight_'+gate_name+'_r')
                return (conv_layer(input_=input_, 
                                   filter_shape=[self.filter_shape[0], self.filter_shape[1], self.input_ch, self.output_ch],
                                   stride_shape=self.stride_shape,
                                   padding=self.padding,
                                   name=gate_name+'_i') +
                        conv_layer(input_=output_prev, 
                                   filter_shape=[self.filter_shape[0], self.filter_shape[1], self.output_ch, self.output_ch],
                                   stride_shape=self.stride_shape,
                                   padding=self.padding,
                                   name=gate_name+'_o') +
                        #tf.einsum('ijkl,jkl->ijkl', read_prev, W))
                        conv_layer(input_=read_prev, 
                                   filter_shape=[self.filter_shape[0], self.filter_shape[1], self.output_ch, self.output_ch],
                                   stride_shape=self.stride_shape,
                                   padding=self.padding,
                                   name=gate_name+'_r'))
                
            # input, forget, and output gates for LSTM
            i = tf.sigmoid(new_gate('input')) # [batch_size, output_height, output_width, output_ch]
            f = tf.sigmoid(new_gate('forget')) # [batch_size, output_height, output_width, output_ch]
            o = tf.sigmoid(new_gate('output')) # [batch_size, output_height, output_width, output_ch]
            update = tf.tanh(new_gate('update')) # [batch_size, output_height, output_width, output_ch]
            # update the sate of the LSTM cell
            hidden = tf.add_n([f * hidden_prev, i * update])
            output = o * tf.tanh(hidden)

            return output, hidden
    # ================================================================================================================
    
    # ===================================================== Build Memory =====================================================
    def build_memory(self, M_prev, read_w_prev, write_w_prev, output):
        """
        M_prev, read_w_prev, write_w_prev, output: have been reshaped
        """
        with tf.variable_scope("memory"):
            # Reading
            if self.read_head_size == 1:
                read_w, read = self.build_read_head(M_prev, read_w_prev, output, 0)
            else:
                # =================================
                # TODO: read head size more than 1=
                # =================================
                pass
            # Writing
            if self.write_head_size == 1:
                write_w, erase, add = self.build_write_head(M_prev, write_w_prev, output, 0)
                
                # M_prev : [batch_size x mem_height x mem_width x mem_size x output_ch]
                # write_w: [batch_size x mem_size x output_ch]
                # erase  : [batch_size x mem_height x mem_width x output_ch]
                # add    : [batch_size x mem_height x mem_width x output_ch]
                M = tf.einsum('ijkl,iml->ijkml', add, write_w) + tf.multiply(M_prev, (1 - tf.einsum('ijkl,iml->ijkml', erase, write_w)))
                
            else:
                # ==================================
                # TODO: write head size more than 1=
                # ==================================
                pass
            
        return M, read_w, write_w, read
            
    def build_read_head(self, M_prev, read_w_prev, output, idx):
        return self.build_head(M_prev, read_w_prev, output, True, idx)
    def build_write_head(self, M_prev, write_w_prev, output, idx):
        return self.build_head(M_prev, write_w_prev, output, False, idx)
    def build_head(self, M_prev, w_prev, output, is_read, idx):
        """
        M_prev: [batch_size x mem_height x mem_width x mem_size x output_ch]
        w_prev: [batch_size x mem_size x output_ch]
        output: [batch_size x output_height x output_width x output_ch]
        """
        scope = "read" if is_read else "write"
        
        with tf.variable_scope(scope):
            # Key
            with tf.variable_scope("k"):
                k = tf.tanh(conv_layer(input_=output, 
                                       filter_shape=[self.filter_shape[0], self.filter_shape[0], self.output_ch, self.output_ch],
                                       stride_shape=self.stride_shape,
                                       padding=self.padding,
                                       name='k_%s' % idx)) # [batch_size x h x w x output_ch]
            # Interpolation gate
            with tf.variable_scope("g"):
                g = tf.sigmoid(tf.squeeze(tensor_linear(input_=output, output_size=1, name='g_%s' % idx))) # [batch_size x output_ch]
            # Shift weighting
            with tf.variable_scope("s_w"):
                w = tensor_linear(input_=output, output_size=3, name='s_w_%s' % idx)
                s_w = tf.nn.softmax(w, dim=1) # [batch_size x shift x output_ch]
            # Sharpen
            with tf.variable_scope("beta"):
                beta  = tf.nn.softplus(tf.squeeze(tensor_linear(output, output_size=1, name='beta_%s' % idx))) # [batch_size x output_ch]
            # Resharpen
            with tf.variable_scope("gamma"):
                gamma = tf.add(tf.nn.softplus(tf.squeeze(tensor_linear(output, output_size=1, name='gamma_%s' % idx))), 
                               tf.constant(1.0)) + 1 # [batch_size x output_ch]
            
            # Consine similarity
            similarity = tensor_cosine_similarity(M_prev, k) # [batch_size x mem_size x output_ch]
            # Focusing by content
            content_focused_w = tf.nn.softmax(tf.einsum('ijk,ik->ijk', similarity, beta), dim=1) # [batch_size x mem_size x output_ch]
            
            # Focusing by content [batch_size x mem_size x output_ch]
            gated_w = tf.add_n([
                tf.einsum('ijk,ik->ijk', content_focused_w, g), 
                tf.einsum('ijk,ik->ijk', w_prev, tf.ones(shape=[self.batch_size, self.output_ch])-g)]) 
            
            
            # Convolutional shifts       
            conv_w = tensor_circular_convolution(gated_w, s_w) # [batch_size x mem_size x output_ch]
            
            # Sharpening
            sharp_w = tf.pow(conv_w, 
                             tf.mul(tf.expand_dims(gamma, axis=1), 
                                    tf.ones(shape=[self.batch_size, self.mem_size, self.output_ch], dtype=tf.float32)))
            sharp_w = tf.div(sharp_w, tf.expand_dims(tf.einsum('ijk->ik', sharp_w)+1e-6, axis=1)) # [batch_size x mem_size x output_ch]
            
            if is_read:
                return sharp_w, tf.einsum('ijklm,ilm->ijkm', M_prev, sharp_w) # [batch_size x output_h x output_w x output_ch]
            else:
                # [batch_size x mem_height x mem_width x output_ch]
                erase = conv_layer(input_=output, 
                                   filter_shape=[self.filter_shape[0], self.filter_shape[1], self.output_ch, self.output_ch],
                                   stride_shape=self.stride_shape,
                                   padding=self.padding,
                                   name='erase_%s' % idx) # [batch_size x mem_height x mem_width x output_ch] 
                add   = conv_layer(input_=output, 
                                   filter_shape=[self.filter_shape[0], self.filter_shape[1], self.output_ch, self.output_ch],
                                   stride_shape=self.stride_shape,
                                   padding=self.padding,
                                   name='add_%s' % idx) # [batch_size x mem_height x mem_width x output_ch] 
                return sharp_w, erase, add
    # ========================================================================================================================
    
    # ===================================================== Reshape =====================================================
    # To reshape the input or controller units to a 4-way tensor [batch, height, width, channel]
    # To reshape the output to a 2-way tensor [batch, (height x width x channel)]
    def reshape(self, input_, reshape_type):
        if reshape_type=="input":
            return self.reshape_input(input_)
        elif reshape_type=="output":
            return self.reshape_output(input_)
        elif reshape_type=="controller":
            return self.reshape_controller(input_)
        elif reshape_type=="memory":
            return self.reshape_memory(input_)
        elif reshape_type=="head":
            return self.reshape_head(input_)
    def reshape_input(self, input_):
        return tf.reshape(input_, [self.batch_size, self.input_height, self.input_width, self.input_ch])
    def reshape_output(self, input_):
        return tf.reshape(input_, [self.batch_size, -1])
    def reshape_controller(self, input_):
        return tf.reshape(input_, [self.batch_size, self.output_height, self.output_width, self.output_ch])
    def reshape_memory(self, input_):
        return tf.reshape(input_, [self.batch_size, self.mem_height, self.mem_width, self.mem_size, self.output_ch])
    def reshape_head(self, input_):
        return tf.reshape(input_, [self.batch_size, self.mem_size, self.output_ch])
    # ===================================================================================================================
    
class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_height, input_width, input_ch, output_height, output_width, output_ch, filter_shape, 
                 stride_shape=[1, 1, 1, 1], padding="SAME"):
        # initialize configs
        # input size
        self.input_height = input_height
        self.input_width  = input_width
        self.input_ch     = input_ch
        # output size
        self.output_height = output_height
        self.output_width  = output_width
        self.output_ch     = output_ch
        # filter, stride, padding
        self.filter_shape = filter_shape
        self.stride_shape = stride_shape
        self.padding = padding
        
        # initialization:
        self.__call__(tf.zeros([1, self.input_height*self.input_width*self.input_ch]), self.zero_state(1, dtype=tf.float32))
    @property
    def input_size(self):
        return self.input_height*self.input_width*self.input_ch

    @property
    def output_size(self):
        return self.output_height*self.output_width*self.output_ch

    @property
    def state_size(self):
        return (self.output_height*self.output_width*self.output_ch*2)
    
    def __call__(self, input_, state=None, scope=None):
        """
        input_: of shape [batch_size, state_size=(input_height x input_width x input_ch)]
        output: of shape [batch_size, state_size=(output_height x output_width x output_ch)]
        """
        #self.batch_size = state.get_shape()[0]
        self.batch_size = tf.shape(state)[0]
        input_ = self.reshape(input_, "input")
        
        # Extract hidd_prev and cell_prev
        start_idx = 0
        hidd_prev = tf.slice(state, [0, start_idx], [-1, self.output_height*self.output_width*self.output_ch])
        start_idx += self.output_height*self.output_width*self.output_ch
        cell_prev = tf.slice(state, [0, start_idx], [-1, -1])
        hidd_prev = self.reshape(hidd_prev, reshape_type="output") # [batch_size, out_h, out_w, out_ch]
        cell_prev = self.reshape(hidd_prev, reshape_type="output") # [batch_size, out_h, out_w, out_ch]
        
        # ConvLSTM
        def new_gate(gate_name):
            return (conv_layer(input_=input_, 
                               filter_shape=[self.filter_shape[0], self.filter_shape[1], self.input_ch, self.output_ch],
                               stride_shape=self.stride_shape,
                               padding=self.padding,
                               name=gate_name+"_input") +
                    conv_layer(input_=hidd_prev, 
                               filter_shape=[self.filter_shape[0], self.filter_shape[1], self.output_ch, self.output_ch],
                               stride_shape=self.stride_shape,
                               padding=self.padding,
                               name=gate_name+"_hid") +
                    conv_layer(input_=cell_prev, 
                               filter_shape=[self.filter_shape[0], self.filter_shape[1], self.output_ch, self.output_ch],
                               stride_shape=self.stride_shape,
                               padding=self.padding,
                               name=gate_name+"_cell"))
        
        # input, forget, and output gates for LSTM
        i = tf.sigmoid(new_gate('input')) # [batch_size, output_height, output_width, output_ch]
        f = tf.sigmoid(new_gate('forget')) # [batch_size, output_height, output_width, output_ch]
        o = tf.sigmoid(new_gate('output')) # [batch_size, output_height, output_width, output_ch]
        update = tf.tanh(new_gate('update')) # [batch_size, output_height, output_width, output_ch]
        # update the sate of the LSTM cell
        cell = tf.add_n([f * hidd_prev, i * update])
        hidd = o * tf.tanh(cell) 
        
        # Output
        hidd = self.reshape(hidd, reshape_type="squeeze")
        cell = self.reshape(cell, reshape_type="squeeze")
        state_next = tf.concat(1, [hidd, cell])
        
        return hidd, state_next
    
    # ===================================================== Reshape =====================================================
    # To reshape the input or controller units to a 4-way tensor [batch, height, width, channel]
    # To reshape the output to a 2-way tensor [batch, (height x width x channel)]
    def reshape(self, input_, reshape_type):
        if reshape_type=="input":
            return self.reshape_input(input_)
        elif reshape_type=="output":
            return self.reshape_output(input_)
        elif reshape_type=="controller":
            return self.reshape_controller(input_)
        elif reshape_type=="memory":
            return self.reshape_memory(input_)
        elif reshape_type=="head":
            return self.reshape_head(input_)
        elif reshape_type=="squeeze":
            return self.reshape_squeeze(input_)
    def reshape_input(self, input_):
        return tf.reshape(input_, [self.batch_size, self.input_height, self.input_width, self.input_ch])
    def reshape_output(self, input_):
        return tf.reshape(input_, [self.batch_size, self.output_height, self.output_width, self.output_ch])
    def reshape_squeeze(self, input_):
        return tf.reshape(input_, [self.batch_size, self.output_height*self.output_width*self.output_ch])
    def reshape_controller(self, input_):
        return tf.reshape(input_, [self.batch_size, self.output_height, self.output_width, self.output_ch])
    def reshape_memory(self, input_):
        return tf.reshape(input_, [self.batch_size, self.mem_height, self.mem_width, self.mem_size, self.output_ch])
    def reshape_head(self, input_):
        return tf.reshape(input_, [self.batch_size, self.mem_size, self.output_ch])
    # ===================================================================================================================    

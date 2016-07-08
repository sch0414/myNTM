from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops

from utils import *
from ops import *

class NTMCell(object):
    def __init__(self, input_dim, output_dim,
            mem_size=128, mem_dim = 20, controller_dim = 100,
            controlley_layer_size = 1, shift_range = 1,
            write_head_size=1, read_head_size=1):
        """Initialize the parameters for an NTM cell.
        Args :
            input_dim : int, The number of units in the LSTM cell
            output_dim : int, The dimensionality of the inputs into the LSTM cell
            mem_size : (optional) int, The size of memory [128]
            mem_dim : (optional) int, The dimensionality for memory [20]
            controller_dim : (optional) int, The dimensionality for controller [100]
            controller_layer_size: (optional) int, The size of controller layer [1]
            """

        # initialize configs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.controller_dim = controller_dim
        self.controller_layer_size = controller_layer_size
        self.shift_range = shift_range
        self.write_head_size = write_head_size
        self.read_head_size = read_head_size

        self.depth = 0
        self.states = []
        
    def __call__(self, input_, state=None, scope=None):
        """Run one step of NTM.

        Args:
            inputs : input Tensor, 2D, 1 * input_size
            state : state Dictionary which contains M, read_w, write_w, read, output, hidden.
            scope : Variable Scope for the created subgrapf; defaults to class name.

        Returns : 
            A tuple containing : 
                - A 2D, batch x output_dim, Tensor representing the outpt of the LSTM after reading "input_" when previous state was "state".
                Here output_dim is :
                    num_proj if num_proj was set,
                    num_units otherwise.
                - A 2d, batch x state_size, Tensor representing the new state of LSTM after reading "input_"when previous state was "state"
                """
        if state == None:
            _, state = self.initial_state()
        
        M_prev = state['M']
        read_w_list_prev = state['read_w']
        write_w_list_prev = state['write_w']
        read_list_prev = state['read']
        output_list_prev = state['output']
        hidden_list_prev = state['hidden']

        # build a controller
        output_list, hidden_list = self.build_controller(input_,
                read_list_prev, output_list_prev, hidden_list_prev)
        
        # last output layer from LSTM controller
        last_output = output_list[-1]

        # build a memory
        M, read_w_list, write_w_list, read_list = self.build_memory(M_prev,
                 read_w_list_prev, write_w_list_prev, last_output)

        # get a new output
        new_output = self.new_output(last_output)

        state = {
                'M' : M,
                'read_w' : read_w_list,
                'write_w' : write_w_list,
                'read' : read_list,
                'output' : output_list,
                'hidden' : hidden_list,
        }
        
        self.depth += 1
        self.states.append(state)
        
        return new_output, state

    def new_output(self, output):
        """Logistic sigmoid output layers"""
        
        with tf.variable_scope('output')
        return tf.sigmoid(Linear(output, self.output_dim, name = 'output'))

    def build_controller(self, input_,
            read_list_prev, output_list_prev, hidden_list_prev):
        """Build LSTM controller."""

        with tf.variable_scope("controller"):
            output_list = []
            hidden_list = []
            for layer_idx in xrange(self.controller_layer_size):
                o_prev = output_list_prev[layer_idx]
                h_prev = hidden_list_prev[layer_idx]

                if layer_idx == 0:
                    def new_gate(gate_name):
                        return linear([input_, o_prev] + read_list_prev,
                                output_size = self.controller_dim,
                                bias = True,
                                scope = "%s_gate_%s" % (gate_name, layer_idx))
                else :
                    def new_gate(gate_name):
                        return linear([output_list[-1], o_prev],
                                output_size = self.controller_dim,
                                bias = True,
                                scope = "%s_gate_%s" % (gate_name, layer_idx))
                # input, forget, and output gates for LSTM
                i = tf.sigmoid(new_gate('input'))
                f = tf.sigmoid(new_gate('forget'))
                o = tf.sigmoid(new_gate('output'))
                update = tf.tanh(new_gate('update'))

                # update the sate of the LSTM cell
                hid =  tf.add_n([f*h_prev, i * update])
                out = o * tf.tanh(hid)
                
                hidden_list.append(hid)
                output_list.append(out)
            
            return output_list, hidden_list

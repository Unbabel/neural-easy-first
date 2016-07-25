'''
Neural Easy First model code
'''

import os
import theano
import theano.tensor as TT
import numpy as np
import sys

from ipdb import set_trace

# local
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/tools')
import shared
from cnn import splice
import gru as rnn
import misc 

def attention(H, S, conv_dim, W_hz, W_sz, w_z, v, name=None):
    '''
    attention_dim  int
    sketch_dim     int 
    conv_dim       int    convolution (splice) total span size

    H              () 
    S              ()
    '''

    # Linear 
    linear = TT.dot(W_hz, splice(H, conv_dim)) + TT.dot(W_sz, splice(S, conv_dim))
    # Non-linear
    z = TT.dot(v, TT.tanh(linear))
    a = TT.nnet.softmax(z)

    # Naming
    name = ('%s.attention' % name) if name else 'attention'
    # Param
    #W_hz.name = '%s.W_hz' % name 
    #W_sz.name = '%s.W_sz' % name 
    #w_z.name  = '%s.w_z' % name 
    #v.name = '%s.v' % name 
    # Vars
    linear.name = '%s.linear' % name
    z.name = '%s.z' % name
    a.name = '%s.a' % name

    return a.T

def sketch_update(H, conv_dim,W_hz, W_sz, w_z, v, W_hs, W_ss, w_s, name=None, S=None):
    '''
    attention_dim  int
    sketch_dim     int 
    conv_dim       int    convolution (splice) total span size

    H              () 
    S              ()
    '''
   
    if S is None:
        S = TT.zeros_like(H)    

    # Attention
    a = attention(H, S, conv_dim, W_hz, W_sz, w_z, v)
    # Signals
    hat_h = TT.dot(H, a)
    hat_s = TT.dot(splice(S, conv_dim), a)
    # Cumulative sketch
    s = TT.tanh(TT.dot(W_hs, hat_h) + TT.dot(W_ss, hat_s) + w_s)
    new_S = S + TT.outer(s, a)
    
    # Naming
    hat_h.name = '%s.hat_h' % name
    hat_s.name = '%s.hat_s' % name
    s.name = '%s.s' % name
    new_S.name = '%s.new_S' % name

    return new_S

def inference(S_m1, sketch_dim, nr_classes, name=None):

    # param
    W_sp = shared.random((nr_classes, sketch_dim))
    w_p = shared.zeros((nr_classes, 1), broadc=(False, True))

    # Linear+Non-Linear
    linear = TT.dot(W_sp, S_m1) + w_p
    p = TT.nnet.softmax(linear.T).T

    # Naming
    name = ('%s.inference' % name) if name else 'inference'
    # Params 
    W_sp.name = '%s.W_sp' % name
    w_p.name = '%s.w_p' % name
    # Variables
    p.name = '%s.p' % name

    return p

def easy_first(H, nr_sketch, attention_dim, sketch_dim, conv_dim, nr_classes, 
               name=None):

    # param attention
    W_hz = shared.random((attention_dim, sketch_dim*conv_dim), name='attention.W_hz')
    W_sz = shared.random((attention_dim, sketch_dim*conv_dim), name='attention.W_sz')
    w_z = shared.zeros((attention_dim, 1), broadc=(False, True), name='attention.w_z')
    v = shared.ones((1, attention_dim), broadc=(True, False), name='attention.v')

    # param sketch update
    W_hs = shared.random((sketch_dim, sketch_dim), name='sketch.W_hs')
    W_ss = shared.random((sketch_dim, sketch_dim*conv_dim), name='sketch.W_ss')
    w_s = shared.zeros((sketch_dim, 1), broadc=(False, True), name='sketch.W_s')
 
    S0 = TT.zeros_like(H)
    S0.name = 'S0'
    sketches = [S0]
    for n in range(nr_sketch):
        new_S = sketch_update(H, conv_dim, W_hz, W_sz, w_z, v, W_hs, W_ss, w_s, S=sketches[-1])
        new_S.name = 'S%d' % (n+1)
        sketches += [new_S]

    p = inference(sketches[-1], sketch_dim, nr_classes)  

    # Naming
    name = ('%s.easy_first' % name) if name else 'easy_first'
    # no params right now
        
    return p

class NeuralEasyFirst():

    def __init__(self, nr_classes=2, emb_matrices=None, model_path=None):

        # config par
        conv_dim = 3  
        attention_dim = 10 
        nr_sketch = 1
        sketch_dim = 20

        # Optimization
        lrate = 0.01

        # QUETCH Layer
        if emb_matrices:

            hidden_size = 20

            E_src, E_trg = emb_matrices
            emb_size_trg, voc_size_trg = E_trg.shape

            # Embeddings
            emb_size_src, voc_size_src = E_src.shape
            E_src = shared.random(size=(emb_size_trg, voc_size_trg), 
                                  name='embeddings_src')
            emb_size_trg, voc_size_trg = E_trg.shape
            E_trg = shared.random(size=(emb_size_trg, voc_size_trg), 
                                  name='embeddings_trg')

            # Input
            x = TT.matrix('x')

            # Quetch layer
            z1 = TT.concatenate((
                E_trg[:, TT.cast(x[0, :], 'int32')],
                E_trg[:, TT.cast(x[1, :], 'int32')],
                E_trg[:, TT.cast(x[2, :], 'int32')],
                E_src[:, TT.cast(x[3, :], 'int32')],
                E_src[:, TT.cast(x[4, :], 'int32')],
                E_src[:, TT.cast(x[5, :], 'int32')]))
            z1_size = emb_size_src*3 + emb_size_trg*3    
            W = shared.random(size=(sketch_dim, z1_size), name='W')
            H = TT.dot(W, z1)

            # GRU
            #param_GRU = gru.init_gru(emb_size, sketch_dim, bias=True)
            #H = gru.gru(emb.T, param_GRU).T
        else:
            x = TT.matrix('H')
            H = x

        #set_trace()
        #dbg = theano.function([x], H)
        #dbg(np.tile(np.arange(20),(6, 1)))

        # Forward 
        if nr_sketch > 1:
            # Easy frist
            p_y = easy_first(H, nr_sketch, attention_dim, sketch_dim, 
                             conv_dim, nr_classes)
        else:
            # Standard quetch
            W2 = shared.random(size=(nr_classes, sketch_dim), name='W2')
            b2 = shared.zeros(size=(1, nr_classes), name='b', 
                              broadc=(True, False))
            z3 = TT.dot(W2, H) #+ b2
            p_y = TT.nnet.softmax(z3)

        print "Compiling forward pass"                 
        self._forward = theano.function([x], p_y)

        # Get params from forward
        self.param = misc.get_param(p_y, named_only=True)

        # SGD Batch update
        y = TT.ivector('y')
        cost = -TT.mean(TT.log(p_y)[y, TT.arange(y.shape[0])])
        updates = [(par, par - lrate*TT.grad(cost, par)) for par in self.param]
        print "Compiling batch update"                 
        self._batch_update = theano.function([x, y], cost, updates=updates)
                 
    def predict(self, input_feat):
        return np.argmax(self._forward(input_feat), 0)    

    def batch_update(self, input_feat, target):
        return self._batch_update(input_feat, target)

    def save(self, model_path):
        raise NotImplementedError()
        pass

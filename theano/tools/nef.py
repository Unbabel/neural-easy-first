'''
Neural Easy First model code
'''

import theano
import theano.tensor as TT
import numpy as np
# local
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

'''
Primitives to handle theano shared variables. These are stored in the GPU 
'''

# Must be installed
import numpy as np
import theano
import theano.tensor as T

def random(size, init=None, rng=None, shared=True, name=None):
    '''
    Fast creation of a shared variable containing an intialized weight matrix
    for a dnn
    '''
    if rng is None:
        rng = np.random.RandomState(1234)

    if init == 'glorot-sigmoid':
        # Numpy
        n_out, n_in = size
        w0          = np.sqrt(6./(n_in + n_out))
        W           = 4*rng.uniform(low=-w0, high=w0, size=size)
    elif init == 'glorot-tanh':
        # Numpy
        n_out, n_in = size
        w0          = np.sqrt(6./(n_in + n_out))
        W           = rng.uniform(low=-w0, high=w0, size=size)
    elif init and ('scale' in init):
        w0 = init['scale']
        W  = rng.uniform(low=-w0, high=w0, size=size)
    else:
        w0 = 0.01
        W  = rng.uniform(low=-w0, high=w0, size=size)
    # Cast
    W = W.astype(theano.config.floatX)
    if shared:
        W = theano.shared(W, borrow=True)
        if name:
            W.name = name
        return W
    else:
        return W

def array(W, name=None, broadc=None, dtype=theano.config.floatX):

    # Shared vars 
    W    = np.array(W).astype(dtype)
    size = W.shape
    # If no broadcast defined try to guess it (this is not Pythonic)
    # TODO: Remove this
    if broadc is not None:
        broadcastable=broadc
    elif size == (1, 1):
        broadcastable=(True, True)
    elif size[0] == 1:
        broadcastable=(True, False)
    elif size[1] == 1:
        broadcastable=(False, True)
    else:
        broadcastable=(False, False)
    W = theano.shared(W, borrow=True, broadcastable=broadcastable)    
    if name:
        W.name = name    
    return W

def zeros(size, broadc=None, dtype=theano.config.floatX, name=None):
    # Numpy
    W = np.zeros(size)
    return array(W, name=name, dtype=dtype, broadc=broadc) 

def ones(size, broadc=None, dtype=theano.config.floatX, name=None):
    # Numpy
    W = np.ones(size, dtype=dtype)
    return array(W, name=name, dtype=dtype, broadc=broadc)

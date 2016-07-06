import sys
import os
#
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
# Local code
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import shared as tl

# DEBUG
from ipdb import set_trace

def splice(x, wsize, padd='zeros', valid=False):
    '''
    Return windowed version of a signal, padd with zeros out fo bound values 
    '''

    if wsize == 1:
        return x

    # Ensure odd windowsize 
    if not wsize % 2:
        raise ValueError, "Windowsize must be even"

    # Half span
    ssiz = (wsize-1)/2

    streams = [x]
    if padd == 'zeros':
        for w in np.arange(1, 1 + ssiz):
            padd  = T.zeros_like(x)[:, :w]
            upper = T.concatenate([padd, x[:, 0:-w]], axis=1)
            lower = T.concatenate([x[:, w:], padd], axis=1)
            streams.insert(0, upper)
            streams.append(lower)
    else:
        for w in np.arange(1, 1 + ssiz):
            wpd   = padd[:, :w]
            upper = T.concatenate([wpd, x[:, 0:-w]], axis=1)
            lower = T.concatenate([x[:, w:], wpd], axis=1)
            streams.insert(0, upper)
            streams.append(lower)
    y = T.concatenate(streams, axis=0)

    # Remove invalid part if solicited
    if valid:
        y = y[:, ssiz:-ssiz]

    return y



def cnn(x, param):

    wsize, W = param
    W        = W.dimshuffle((0,'x','x',1))
    padd     = T.zeros_like(x)[:, :(wsize-1)/2]
    px       = T.concatenate((padd, x, padd), axis=1).dimshuffle(('x','x', 0, 1))
    z        = conv.conv2d(px, W)
    y        = T.reshape(z[0, :, :, :], (z.shape[1]*z.shape[2], z.shape[3]), ndim=2)
    return y 

if __name__ == '__main__':

    data_x = np.random.randn(5, 3).astype(theano.config.floatX)

    wsize = 3
    W     = tl.random((1, wsize)).dimshuffle((0,'x','x',1)) 

    x    = T.matrix('x')
    padd = T.zeros_like(x)[:, :(wsize-1)/2] 
    px   = T.concatenate((padd, x, padd), axis=1).dimshuffle(('x','x', 0, 1))
    z    = conv.conv2d(px, W)

    dbg = theano.function([x], z)

    set_trace()
    print "" 

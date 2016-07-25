import sys
import os
# Must be installed
import numpy as np
import theano
import theano.tensor as T
# Local code
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import shared as tl

# DEBUG
#from ipdb import set_trace

def init_gru(input_size, hid_size, bias=False, name=None): 
    '''
    Most common GRU initialization

    NOTE: The bias are allways initizalized for gru unfolding in axis 0 
    '''

    # Input-Hidden linear transformations
    W   = tl.random((hid_size, input_size), init='glorot-tanh', shared=False)
    W_r = tl.random((hid_size, input_size), init='glorot-sigmoid', shared=False)
    W_z = tl.random((hid_size, input_size), init='glorot-sigmoid', shared=False)
    # concatenate all linear transf [W, W_r, W_z]
    W  = tl.array(np.concatenate((W, W_r, W_z)))
    # Naming
    name = ('%s.gru' % name) if name else 'gru'
    W.name = '%s.[W, W_r, W_z]' % name

    # Hidden-Hidden linear transformations
    U   = tl.random((hid_size, hid_size), init='glorot-tanh')
    U_r = tl.random((hid_size, hid_size), init='glorot-sigmoid')
    U_z = tl.random((hid_size, hid_size), init='glorot-sigmoid')
    # Naming
    U.name   = '%s.U' % name
    U_r.name = '%s.U_r' % name
    U_z.name = '%s.U_z' % name

    # Append
    param = [W, U, U_r, U_z] 

    if bias:
        # Input-Hidden Bias
        b_W       = tl.zeros((1, 3*hid_size), broadc=(True, False))
        b_W.name  = '%s.[b b_r b_z]' % name
        param    += [b_W]

        # Hidden-Hidden bias
        b_U  = tl.zeros((1, hid_size), broadc=(True, False))
        b_Ur = tl.zeros((1, hid_size), broadc=(True, False))
        b_Uz = tl.zeros((1, hid_size), broadc=(True, False))
        # Naming
        b_U.name  = '%s.b_U' % name
        b_Ur.name = '%s.b_Ur' % name       
        b_Uz.name = '%s.b_Uz' % name   
        # Append
        param += [b_U, b_Ur, b_Uz]

    return param

def gru(z1, param, h0=None):
    '''
    NOTE: It unfolds on axis 0! use transpose if you want the opposite
    '''

    # Argument handling
    if len(param) == 8:
        # Uses bias
        W, U, U_r, U_z, b_W, b_U, b_Ur, b_Uz = param
        bias = True
    elif len(param) == 4: 
        W, U, U_r, U_z = param
        bias = False 
    else:
        raise Exception, "Expected 8 or 4 parameters for the GRU"

    # Hidden dimension
    hid_size = param[0].get_value().shape[0]/3

    # Initial state
    if h0 is None:
        h0 = tl.zeros((1, hid_size), broadc=(False, False))

    # Unfolded GRU transformation
    if bias:

        # The linear projection can be done outside of scan
        hx = T.dot(z1, W.T) + b_W 

        def gru_step(hx_tm1, h_tm1):
            # reset gate
            r       = T.nnet.sigmoid(hx_tm1[hid_size:2*hid_size] 
                                     + T.dot(h_tm1, U_r.T) + b_Ur)
            # candidate activation
            tilde_h = T.tanh(hx_tm1[:hid_size] + T.dot(r*h_tm1, U.T) + b_U) 
            # update gate
            z = T.nnet.sigmoid(hx_tm1[2*hid_size:3*hid_size] 
                               + T.dot(h_tm1, U_z.T) + b_Uz)
            return (1-z)*h_tm1 + z*tilde_h

    else:

        # The linear projection can be done outside of scan
        hx = T.dot(z1, W.T)

        def gru_step(hx_tm1, h_tm1):
            # reset gate
            r       = T.nnet.sigmoid(hx_tm1[hid_size:2*hid_size] 
                                     + T.dot(h_tm1, U_r.T))
            # candidate activation
            tilde_h = T.tanh(hx_tm1[:hid_size] + T.dot(r*h_tm1, U.T)) 
            # update gate
            z = T.nnet.sigmoid(hx_tm1[2*hid_size:3*hid_size] 
                               + T.dot(h_tm1, U_z.T))
            return (1-z)*h_tm1 + z*tilde_h

    # This creates the variable length computation graph (unrols the rnn)
    h, updates = theano.scan(fn=gru_step, 
                             sequences=hx, 
                             outputs_info=dict(initial=h0))

    # Remove intermediate empty dimension
    return h[:,0,:]

def init_bigru(input_size, hid_size, bias=False, name=None): 

    # Forward parameters
    param_f = init_gru(input_size, hid_size, bias=bias)
    # Naming
    name = ('%s.bigru' % name) if name else 'bigru'
    param_f[0].name = '%s.[Wf, W_rf, W_zf]' % name
    param_f[1].name   = '%s.Uf' % name
    param_f[2].name = '%s.U_rf' % name
    param_f[3].name = '%s.U_zf' % name

    # Backward parameters
    param_b = init_gru(input_size, hid_size, bias=bias)
    # Naming
    param_b[0].name = '%s.[Wb, W_rb, W_zb]' % name
    param_b[1].name = '%s.Ub' % name
    param_b[2].name = '%s.U_rb' % name
    param_b[3].name = '%s.U_zb' % name

    return param_f + param_b

def bigru(x, param, hf0=None, hb0=None):
    '''
    Bi-directional rnn, see rnn
    '''

    # Argument handling
    if len(param) == 16:
        # Uses bias
        bias = True
    elif len(param) == 8: 
        bias = False 
    else:
        raise Exception, "Expected 16 or 8 parameters for the GRU"

    param_f = param[:len(param)/2]
    param_b = param[len(param)/2:]

    # Initial state
    if hf0 is None:
        hid_size = param_f[0].get_value().shape[0]/3
        hf0 = tl.zeros((1, hid_size), broadc=(False, False))

    # Forward GRU 
    hf = gru(x, param_f, h0=hf0)

    # Start with last hidden state
    if hb0 == 'reverse':
        hb0 = hf[-1:, :]
    elif hb0 is None:
        hid_size = param[0].get_value().shape[0]/3
        hb0 = tl.zeros((1, hid_size), broadc=(False, False))

    # Backward GRU 
    hb = gru(x[::-1, :], param_b, h0=hb0)[::-1, :]

    return hf, hb


if __name__ == '__main__':

    # For debug
    theano.config.optimizer='fast_compile'   # 'None'

    # ARGUMENT HANDLING
    tests = ['gru', 'bigru', 'DAE', 'all']
    if len(sys.argv) != 2 or sys.argv[1] not in tests:
        print "\nUnit tests: %s\n" % (" ".join(tests))
        exit(1)
    test = sys.argv[1]    

    # FAKE DATA
    voc_size  = 100
    sent_size = 7
    probs     = [1/voc_size]*voc_size
    data_x    = np.nonzero(np.random.multinomial(1, probs, size=sent_size))[1]
    data_x    = data_x.astype('int32')

    # SWTCH OVER UNIT-TESTS
    if test == "gru" or test == "all":
       
        # MODEL
        input_size = 20
        hid_size = 10
        # Embeddings
        E = tl.random((input_size, voc_size), init='glorot-sigmoid') 
        
        # FORWARD
        idx = T.ivector()
        # Embedding
        z1  = E[:, idx]

        # GRU
        param = init_gru(input_size, hid_size, bias=True)
        z2    = gru(z1.T, param).T
    
        print "Compiling GRU" 
        dbg = theano.function([idx], z2)

        print dbg(np.array([1, 3, 4, 1, 8]).astype('int32'))

        print "UNIT TEST GRU ...",
        print "\033[32mOK\033[0m"

    if test == "bigru" or test == "all":

        # MODEL
        input_size = 20
        hid_size = 10
        # Embeddings
        E = tl.random((input_size, voc_size), init='glorot-sigmoid') 
        
        # FORWARD
        idx = T.ivector()
        # Embedding
        z1  = E[:, idx]

        # BI-GRU        
        param = init_bigru(input_size, hid_size, bias=True)
        zf2, zb2 = bigru(z1.T, param)

        print "Compiling BI-GRU" 
        dbg = theano.function([idx], [zf2.T, zb2.T])

        print dbg(np.array([1, 3, 4, 1, 8]).astype('int32'))

        print "UNIT TEST BI-GRU ...",
        print "\033[32mOK\033[0m"

    if test == "DAE" or test == "all":

        # DATA
        data_x = np.random.randn(3, 10).astype(theano.config.floatX)

        # Config
        input_size = 3
        hid_size   = 3

        # Param
        param1 = init_gru(input_size, hid_size, bias=True)
        param2 = init_gru(input_size, hid_size, bias=True)
        W      = tl.random((input_size, hid_size))

        # Forward
        x = T.matrix('x')
        z1 = gru(x.T, param1).T        
        z2 = gru(z1.T, param2).T        
        hat_x = T.dot(W, z2)

        # Cost
        y    = T.matrix('y')
        cost = T.mean(T.sum((y-hat_x)**2, 0))

        # Batch update
        lrate = 0.01
        param    = param1 + param2 + [W]
        updates  = [(par, par -lrate*T.grad(cost, par)) for par in param]
        batch_up = theano.function([x, y], cost, updates=updates) 

        set_trace()
        print ""

        batch_update(data_x, data_x)

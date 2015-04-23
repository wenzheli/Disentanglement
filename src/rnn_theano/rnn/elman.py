import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class model(object):
    
    def __init__(self, ni, nh, no, cs):
        '''
        ni :: dimension of the input layer
        nh :: dimension of the hidden layer
        no :: dimension of the output layer
        cs :: window size.   
        '''
        # parameters of the model
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ni, nh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, no)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(no, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))  # initial value for state 0

        # bundle
        self.params = [self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        x = T.dmatrix()    # this is the input of size cs * ni 
        y = T.dscalar('y') # actual output value, which is real value.  
        
        # this function represents how to get the next state representation and output. 
        # x_t is the input for time t, and h_tm1 is the vector representation of the hidden state. 
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.dot(h_t, self.W) + self.b
            return [h_t, s_t]
        
        # return all the hidden states and outputs.  h is the matrix
        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])
        
        y_pred = s[-1,:]
        
        # cost and gradients and learning rate
        lr = T.scalar('lr')
        
        nll = T.mean((y_pred - y)**2)
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[x], outputs=y_pred)

        self.train = theano.function( inputs  = [x, y, lr],
                                      outputs = nll,
                                      updates = updates)
        
    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

import numpy 
import time
import sys
import subprocess
import os
import random
import pdb
import csv

from data import process_stock_data
from rnn.elman import model


if __name__ == '__main__':
    
    # store all the parameters....
    s = {'lr':0.0627142536696559,
        'verbose':1,
        'decay':False,   # decay on the learning rate if improvement stops
        'win':20,        # number of words in the context window
        'bs':9,          # number of backprop through time steps
        'nhidden':3,     # number of hidden units
        'seed':345,
        'nepochs':20,
        't_idx' : 0}
    
    # get the data
    train_data, train_label, valid_data, valid_label, test_data, test_label \
        = process_stock_data.get_data(s['t_idx'])
    
    # instanciate the RNN model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(ni = 19,    # we have total 20 stocks, one of which is target time series
                nh = s['nhidden'],
                no = 1,     # the dimension of output is 1, since we have only one target series
                cs = s['win'] )
    
    win_size = s['win']     # window size. 
    
    
    train_errors = []
    valid_errors = []
    test_errors = []
    test_pred = []
    test_actual = []
    valid_pred = []
    valid_actual = []
    best_valid_error = 100
    
    for itr in xrange(s['nepochs']):
        n_train = len(train_data)
        # iterate through the training data, and learn the RNN model. 
        for end_idx in xrange(win_size, n_train):
            data = numpy.asarray(train_data[end_idx - win_size : end_idx]).astype('float64')
            y = numpy.asarray(train_label[end_idx]).astype('float64')
            rnn.train(data, y, s['lr'])
        
        # evaluate on the training data
        train_error = 0
        for end_idx in xrange(win_size, n_train):
            data = numpy.asarray(train_data[end_idx - win_size : end_idx]).astype('float64')
            y = numpy.asarray(train_label[end_idx]).astype('float64')
            pred = rnn.classify(data)[0]
            train_error = train_error + (pred-y)**2
        train_errors.append(train_error / (n_train - win_size))
        
        # evaluate on the validation data
        valid_error = 0
        n_valid = len(valid_data)
        for end_idx in xrange(win_size, n_valid):
            data = numpy.asarray(valid_data[end_idx - win_size : end_idx]).astype('float64')
            y = numpy.asarray(valid_label[end_idx]).astype('float64')
            pred = rnn.classify(data)[0]
            valid_error = valid_error + (pred-y)**2
            valid_pred.append(pred)
            valid_actual.append(y)
        mean_valid_error = valid_error/(n_valid -win_size)
        valid_errors.append(mean_valid_error)
        
        print itr, ' train : ', train_error/(n_train-win_size), ' valid :', valid_error/(n_valid-win_size)\
        
   
    # write the results into the file. 
    f = open('valid_pred','w')
    for val_true, val_pred in zip(valid_actual, valid_pred):
        f.write(str(val_true) + ',' + str(val_pred) + '\n')
    f.close()
      
    f = open('error_graph','w')
    for t_err, v_err in zip(train_errors, valid_errors):
        f.write(str(t_err) + ',' + str(v_err) + '\n')
    f.close()
    
    
    
    
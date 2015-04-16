import csv
import numpy as np

def get_data(t_idx):
    """ 
    It contains stock price data. The most recent price appears first, we need
    to reverse the list.
    t_idx ::  the index for the target time series. i.e if t_idx = 0, we use the 
              first stock as our prediction target. 
    """
    train_data = []
    train_label = []
    valid_data = []
    valid_label = []
    test_data = []
    test_label = []
    
    lines = []
    #f = open('/home/liwenzhe/myworkspace/is13/dataset/closingAdjLog.csv', 'rb')
    f = open('dataset/closingAdjLog.csv')
    reader = csv.reader(f)
    for row in reader:
        lines.append(row)
    lines.pop(0)
    
    n = len(lines)
    # use the first 80% for the training data, the next 10% for the validation data
    # and the last 10% for the test data. 
    # TODO : create folds instead. 
    for i in xrange(n-1, -1, -1):
        curr_line = lines[i]
        if i > n * 0.2:
            # add into training data
            if t_idx == 0:
                train_data.append(curr_line[t_idx+1:])
            else:
                train_data.append(curr_line[:t_idx-1] + curr_line[t_idx:])
            train_label.append(curr_line[t_idx]) 
        elif i > n * 0.1:
            # add into validation data
            if t_idx == 0:
                valid_data.append(curr_line[t_idx+1:])
            else: 
                valid_data.append(curr_line[:t_idx-1] + curr_line[t_idx:])
            valid_label.append(curr_line[t_idx]) 
        else:
            # add into test data
            if t_idx == 0:
                test_data.append(curr_line[t_idx+1:])
            else:
                test_data.append(curr_line[:t_idx-1] + curr_line[t_idx:])
            test_label.append(curr_line[t_idx]) 
    
    return [train_data, train_label, valid_data, valid_label, test_data, test_label]
        
    
    

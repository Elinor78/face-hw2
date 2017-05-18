import csv
import sys
import numpy as np 
from keras.utils import np_utils

def read_csv(filepath):
    '''
    reads data from given csv files
    '''
    d = []
    with open(filepath, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            d.append(row)
    return np.array(d)

def to_cat(data):
    '''
    transforms decimal categorical to matrix categorical
    '''
    return np_utils.to_categorical(data, 3)

def to_submission(data):
    '''
    appends an index to each data point
    '''
    d = []
    for index, item in enumerate(data):
        d.append([index,item])
    return d 

def load_data():
    '''
    reads data from csv, reshapes it
    '''
    #think about reshaping to 1x48x48, if...
    train_data = read_csv('raw_data/train_data.csv')
    train_target = read_csv('raw_data/train_target.csv')
    test_data = read_csv('raw_data/test_data.csv')

    train_target = to_cat(train_target)
    train_data = train_data.reshape(-1, 1, 48, 48)
    test_data = test_data.reshape(-1, 1, 48, 48)

    return train_data, train_target, test_data

def read_class_output_file(filepath):
    '''
    reads single array class output file
    '''
    with open(filepath, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            return row

def write_submission(data, filepath):
    '''
    formats the submission 
    '''
    data = to_submission(data)

    with open(filepath, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Category'])
        for row in data:
            writer.writerow(row)
   
class WrongNumberOfArgs(Exception):
    pass


if __name__ == '__main__':

    if len(sys.argv) != 3:
        raise WrongNumberOfArgs('wrong number of args, need infile and outfile')

    class_output = read_class_output_file(sys.argv[1])
    write_submission(class_output, sys.argv[2])





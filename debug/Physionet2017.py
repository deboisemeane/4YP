#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:02:54 2022

Class for loading data from the Physionet 2017 database. The class is iterable, and takes the following arguments:
    root: The file path to the Physionet 2017 database.
    train: If true, loads the training dataset, otherwise loads the validation dataset
    batch_size: The batch size to load with each iteration, for SGD
    shuffle: Whether to shuffle the database prior to loading batches
    resample: Whether to resample data from 300 Hz to 125 Hz
    filt: Whether to apply a 0.67 Hz 8th order high-pass Butterworth filter
    verbose: Whether to print information to the console

@author: shaun
"""

from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, detrend
import scipy.io
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

class Physionet2017(object):
    
    def __init__(self, root = "/users/shaun/code/atrial_fibrillation/", train = True, batch_size = 4, shuffle = True, resample = False, filt = False, verbose = False):
        if train == True:
            self.root = root + 'training/'
        else:
            self.root = root + 'validation/'
        self.batch_size = batch_size
        self.verbose = verbose
        if resample == True:
            self.sample_rate = 125
        else:
            self.sample_rate = 300
        if filt == True:
            self.filter = True
            [self.b, self.a] = butter(8, 0.67*2/self.sample_rate, btype='high')
        else:
            self.filter = False
        
        with open(self.root + 'REFERENCE.csv', mode = 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            self.reference = []
            for row in csv_reader:
                self.reference.append(row)
                
        if shuffle == True:
            random.shuffle(self.reference)
        
    def load_records(self, index):
        records = np.zeros((self.batch_size, 1, 61*self.sample_rate), dtype = 'int16')
        labels = np.zeros(self.batch_size)
        mask = ['A', 'N', 'O', '~']
        count = 0
        
        for i in range(index, index + self.batch_size):
            if self.verbose:
                print(self.reference[i][0])

            for j in range(0, 4):
                if self.reference[i][1] == mask[j]:
                    labels[count] = j
            
            record = scipy.io.loadmat(self.root + self.reference[i][0])["val"]
            
            if self.sample_rate == 125:
                spline = CubicSpline(np.arange(0,len(record[0])/300,1/300), record, axis=1)
                record = spline(np.arange(0,len(record[0])/300,1/125))
            
            if self.filter == True:
                detrend(record, -1, 'linear', 0, True)
                record = filtfilt(self.b, self.a, record, padtype = 'odd', padlen=3*(max(len(self.b),len(self.a))-1))
                
            records[count, 0, 0:len(record[0])] = record
            count += 1
            
        return torch.from_numpy(records).float(), torch.from_numpy(labels).type(torch.LongTensor)
    
    def __iter__(self):
        return PhysionetIterator(self)

class PhysionetIterator:
    
    def __init__(self, reference):
        self.reference = reference
        self.index = 0
        
    def __next__(self):
        if self.index + self.reference.batch_size <= len(self.reference.reference):
            records, labels = self.reference.load_records(self.index)
            self.index += self.reference.batch_size
            return records, labels
        else:
            raise StopIteration

#Allows you to test the behaviour of the Physionet 2017 data loader class      
if __name__ == '__main__':
    
    #Configure settings
    dataset = Physionet2017("/users/shaun/code/atrial_fibrillation/", False, 4, False, True, True, False)
    plot_batch = 30
    
    #Iterates through the dataset
    batch_count = 0
    for data in dataset:
        records, labels = data
        batch_count += 1
        
        print(batch_count)
        
        #Plots the waveforms in a single batch
        if batch_count == plot_batch:
            mask = ['A', 'N', 'O', '~']
            plt.figure(1)
            for i in range(0, 4):
                plt.plot(np.arange(0,30,1/125), records[i, 0, 0:3750])
            plt.ylabel('ECG Signal')     
            plt.xlabel('Time, s')
            plt.legend(labels)
            break

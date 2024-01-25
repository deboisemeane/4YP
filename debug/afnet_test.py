#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 18:42:39 2022

Tests the performance of a neural net trained on the Physionet 2017 database.
Parameters that may be set by the user include:
    test_all: Give the accuracy on the overall dataset
    test_class: Give the accuracy for each class
    f1_score: Give the F1 score on the overall dataset
    net_name: Which neural net to evaluate
    train_set: Whether to evaluate performance on the training or validation set
    save_mode: Whether to write a results table to file (for docker volume testing)

@author: shaun
"""

import Physionet2017 as Phys
import AFNet7 as AFNet
import torch
import torchmetrics.functional as metrics
import csv #Used for docker volume testing

if __name__ == '__main__':
    #What tests to perform
    test_all = True
    test_class = True
    f1_score = True
    save_mode = True #Used for docker volume testing
    
    #What network to load and what dataset to use
    net_name = './afnet_7_filt'
    train_set = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if train_set:
        batch_size = 8528
    else:
        batch_size = 300
    
    #Loads the dataset
    dataset = Phys.Physionet2017("../", train_set, batch_size, True, True, True, False) #NOTE: Relative path for docker testing

    #Loads the neural network
    net = AFNet.AFNet()
    net.load_state_dict(torch.load(net_name, map_location=device))
    net.eval() #disables the dropout layers in the neural network

    if test_all:
        #Evaluates overall perforamnce
        correct = 0
        total = 0
        
        #Skips gradient calculations (we're not training) for speed
        with torch.no_grad():
            for data in dataset:
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print(f'Accuracy of the network on the {batch_size} recordings is: {100 * correct // total} %')
        
    if test_class:
        #Evaluates performance on individual classes
        classes = ('A', 'N', 'O', '~')
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        
        with torch.no_grad():
            for data in dataset:
                inputs, labels = data
                outputs = net(inputs)
                _, predictions = torch.max(outputs, 1)
                
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
                    
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
            
    if f1_score:
        with torch.no_grad():
            for data in dataset:
                inputs, labels = data
                outputs = net(inputs)
                _, predictions = torch.max(outputs, 1)
        F1 = metrics.f1_score(predictions, labels) #, average=None, num_classes=4)
        print(f'F1 Score is: {F1:.2f}')
        
    #Writes a test file to drive, for docker volume testing
    if save_mode and test_class:
        w = csv.writer(open("../traffic/accuracy.csv", "w"))
        for key, val in correct_pred.items():
            w.writerow([key, val])
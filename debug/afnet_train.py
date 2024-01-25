#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:36:04 2022

Trains a neural net on the Physionet 2017 database. Default settings are to 
train for 50 epochs with a learning rate of 0.01 decreasing by a factor of 0.95
per epoch and using a batch size of 16.

@author: shaun
"""

import Physionet2017 as Phys
import AFNet7 as AFNet #Change the module name to use alternative network architectures
import torch.nn as nn
import torch.optim as optim
import torch

if __name__ == '__main__':
    save_as = './afnet_new'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = AFNet.AFNet()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(50):
        dataset = Phys.Physionet2017("/users/shaun/code/atrial_fibrillation/", True, 16, True, True, True, False)
        
        learn_rate = 0.01*0.95**epoch
        print('Learning Rate = ', learn_rate)
        optimiser = optim.Adam(net.parameters(), lr = learn_rate)
        
        running_loss = 0.0
    
        for i, data in enumerate(dataset, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimiser.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.5f}')
                running_loss = 0.0
                        
    print('Finished Training')

    #Saves the trained neural network
    
    torch.save(net.state_dict(), save_as)
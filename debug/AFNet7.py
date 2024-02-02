#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:14:41 2022

Class for implementing a 7-layer CNN as described in Vogt et al. 2018. 

@author: shaun
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

#Defines the structure of the neural network
class AFNet(nn.Module):
    
    #Initialises the various layers of the network
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d( 1,  16, 21)
        self.conv2 = nn.Conv1d( 16,  32, 21)
        self.conv3 = nn.Conv1d( 32,  32, 21)
        self.conv4 = nn.Conv1d( 32,  64, 21)
        self.conv5 = nn.Conv1d( 64,  64, 21)
        self.conv6 = nn.Conv1d( 64, 128, 21)
        self.conv7 = nn.Conv1d(128, 128, 21)
        
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.4)
        self.drop5 = nn.Dropout(0.5)
        self.drop6 = nn.Dropout(0.5)
        
        self.pool = nn.AvgPool1d(2, 2)
        
        self.norm0 = nn.BatchNorm1d(1)
        self.norm1 = nn.BatchNorm1d(16)
        self.norm2 = nn.BatchNorm1d(32)
        self.norm3 = nn.BatchNorm1d(32)
        self.norm4 = nn.BatchNorm1d(64)
        self.norm5 = nn.BatchNorm1d(64)
        self.norm6 = nn.BatchNorm1d(128)
        self.norm7 = nn.BatchNorm1d(128)
        
        self.fc1 = nn.Linear(128, 4)
        
    # Defines the behaviour of a forward pass on an input
    def forward(self, x):

        # The convolutional layers
        x = f.relu(self.norm1(self.conv1(self.norm0(x))))
        x = self.drop2(f.relu(self.norm2(self.conv2(x))))
        x = self.pool(x)
        x = self.drop3(f.relu(self.norm3(self.conv3(x))))
        x = self.pool(x)
        x = self.drop4(f.relu(self.norm4(self.conv4(x))))
        x = self.pool(x)
        x = self.drop5(f.relu(self.norm5(self.conv5(x))))
        x = self.pool(x)
        x = self.drop6(f.relu(self.norm6(self.conv6(x))))
        x = self.pool(x)
        x = f.relu(self.norm7(self.conv7(x)))
        
        
        #Global max pooling and fully connected layer for classification
        #x = torch.max(x, 2)[0]

        # Trying global average pooling instead
        x = torch.mean(x, 2)  # Averaging over special dimension produces one value per channel.

        x = self.fc1(x)
        #x = f.log_softmax(x, dim=1)

        return x


if __name__ == '__main__':
    x = torch.zeros((64,1,15000))
    model = AFNet()
    y = model(x)
    print(y.shape)

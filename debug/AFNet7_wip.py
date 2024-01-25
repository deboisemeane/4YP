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
class AFNet_wip(nn.Module):
    
    #Initialises the various layers of the network
    def __init__(self):
        super().__init__()
        stride = 4
        self.conv1 = nn.Conv1d(1, 16, 21, 2)
        self.conv2 = nn.Conv1d(16, 32, 21, 2)
        self.conv3 = nn.Conv1d(32, 32, 21, 2)
        self.conv4 = nn.Conv1d(32, 64, 21, 2)
        self.conv5 = nn.Conv1d(64, 64, 21, 2)
        self.conv6 = nn.Conv1d(64, 128, 21, stride)
        self.conv7 = nn.Conv1d(128, 128, 21, stride)

        self.fc1 = nn.Linear(3712, 128)
        self.fc2 = nn.Linear(128, 4)

        # Defines the behaviour of a forward pass on an input

    def forward(self, x):
        # Alex's zero padding to make 15000 into 18300
        x = torch.nn.functional.pad(input=x, pad=(0, 3300), value=0)

        # The convolutional layers
        x = f.relu((self.conv1(x)))
        x = (f.relu((self.conv2(x))))
        x = (f.relu((self.conv3(x))))
        x = (f.relu((self.conv4(x))))
        x = (f.relu((self.conv5(x))))
        x = (f.relu((self.conv6(x))))
        x = f.relu((self.conv7(x)))

        # Global max pooling and fully connected layer for classification
        x = torch.flatten(x, start_dim=1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        # x = f.log_softmax(x, dim=1)

        return x
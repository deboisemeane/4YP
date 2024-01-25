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
        padding7 = 3
        padding5 = 2
        padding3 = 1
        self.conv1 = nn.Conv1d(1, 128, 7, 2, padding=padding7)
        self.conv2 = nn.Conv1d(128, 128, 7, 2, padding=padding7)
        self.conv3 = nn.Conv1d(128, 128, 7, 2, padding=padding7)
        self.conv4 = nn.Conv1d(128, 128, 7, 2, padding=padding5)
        self.conv5 = nn.Conv1d(128, 128, 7, 2, padding=padding5)
        self.conv6 = nn.Conv1d(128, 128, 7, 2, padding=padding7)
        self.conv7 = nn.Conv1d(128, 256, 7, 2, padding=padding7)
        self.conv8 = nn.Conv1d(256, 256, 5, 2, padding=padding5)
        self.conv9 = nn.Conv1d(256, 256, 5, 2, padding=padding5)
        self.conv10 = nn.Conv1d(256, 256, 5, 2, padding=padding3)
        self.conv11 = nn.Conv1d(256, 256, 3, 2, padding=padding3)
        self.conv12 = nn.Conv1d(256, 256, 3, 2)

        self.fc1 = nn.Linear(768, 100)
        self.fc2 = nn.Linear(100, 4)
        
    #Defines the behaviour of a forward pass on an input
    def forward(self, x):
        R = nn.ReLU
        #The convolutional layers
        x = R(self.conv1(x))
        x = R(self.conv2(x))
        x = R(self.conv3(x))
        x = R(self.conv4(x))
        x = R(self.conv5(x))
        x = R(self.conv6(x))
        x = R(self.conv7(x))
        x = R(self.conv8(x))
        x = R(self.conv9(x))
        x = R(self.conv10(x))
        x = R(self.conv11(x))
        x = R(self.conv12(x))
        x = torch.flatten(x, start_dim=1)  # Start at dim1 such that the batch dimension is preserved.
        x = R(self.fc1(x))
        x = self.fc2(x)

        return x


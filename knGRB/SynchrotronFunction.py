# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:23:49 2024

@author: rohdo
"""
import numpy as np

table = np.transpose(np.loadtxt('firstSynchrotronFunction.txt'))

def synF(x):
    """Tabulated first Synchrotron function with asymptotic approximation."""
    
    tableX = table[0]
    tableF = table[1]
    
    # eq (8.65) in Longair 
    asym = np.sqrt(np.pi/2) * np.sqrt(x) * np.exp(-x)
    F = np.interp(x, tableX, tableF, left = 0, right = 0)
    
    return np.where(x <= 100, F, asym)
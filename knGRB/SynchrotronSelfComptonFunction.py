# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:26:13 2024

@author: rohdo
"""

import numpy as np

def gKN(q, w):
    """gKN as defined in Lemoine (17) where 
    w = 1-(1+z)*h*nu/(gammaBulk*gamma*m_e*c**2)
    """
    G = (1 - w)/w
    return 2 * q * np.log(q) + (1 + 2*q) * (1 - q) + G**2 * (1 - q)/(2 * (1 + G))
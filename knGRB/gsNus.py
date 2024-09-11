# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:02:45 2024

@author: rohdo
"""

import numpy as np

def gsNuMslow(param): # TODO: test before implementation
    """The GS01 calculation for slow cooling nuM b=2 in the observer frame.
    
        parameters
        ----------
        param : ConvertParam Object
            The GRB parameters as a ConvertParam Object.
    """
    epsEbar = param.epsE * (param.p - 2)/(param.p - 1)
    
    if param.k == 0:
        return 3.73 * (param.p - 0.67) * 1e15 * (1 + param.z)**(1/2) * (param.blastEnergy * 1e7/1e52)**(1/2) * epsEbar**(2) * param.epsB**(1/2) * (param.tZ * ( 1 + param.z)/86400)**(-3/2)
    elif param.k == 2:
        return 4.02 * (param.p - 0.69) * 1e15 * (1 + param.z)**(1/2) * (param.blastEnergy * 1e7/1e52)**(1/2) * epsEbar**(2) * param.epsB**(1/2) * (param.tZ * ( 1 + param.z)/86400)**(-3/2)
    
def gsNuMfast(param): 
    """The GS01 calculation for fast cooling nuM b=9 in the observer frame.
    
        parameters
        ----------
        param : ConvertParam Object
            The GRB parameters as a ConvertParam Object.
    """
    epsEbar = param.epsE * (param.p - 2)/(param.p - 1)
    
    if param.k == 0:
        return 3.94 * (param.p - 0.74) * 1e15 * (1 + param.z)**(1/2) * (param.blastEnergy * 1e7/1e52)**(1/2) * epsEbar**(2) * param.epsB**(1/2) * (param.tZ * ( 1 + param.z)/86400)**(-3/2)
    if param.k == 2:
        return 3.52 * (param.p - 0.31) * 1e15 * (1 + param.z)**(1/2) * (param.blastEnergy * 1e7/1e52)**(1/2) * epsEbar**(2) * param.epsB**(1/2) * (param.tZ * ( 1 + param.z)/86400)**(-3/2)
    
def gsNuCslow(param): 
    """The GS01 calculation for slow cooling nuC b=3 in the observer frame.
    
        parameters
        ----------
        param : ConvertParam Object
            The GRB parameters as a ConvertParam Object.
    """
    if param.k == 0:
        return 6.37 * (param.p - 0.46) * 1e13 * np.exp(-1.16 * param.p) * (1 + param.z)**(-1/2) * param.epsB**(-3/2) * (param.numDens * 1e-6)**(-1) * (param.blastEnergy * 1e7/1e52)**(-1/2) * (param.tZ * ( 1 + param.z)/86400)**(-1/2)
    elif param.k == 2:
        return 4.40 * (3.45 - param.p) * 1e10 * np.exp(0.45 * param.p) * (1 + param.z)**(-3/2) * param.epsB**(-3/2) * param.Astar**(-2) * (param.blastEnergy * 1e7/1e52)**(1/2) * (param.tZ * ( 1 + param.z)/86400)**(1/2)

def gsNuCfast(param):
    """The GS01 calculation for fast cooling nuC b=11 in the observer frame.
    
        parameters
        ----------
        param : ConvertParam Object
            The GRB parameters as a ConvertParam Object.
    """
    if param.k == 0:
        return 5.86e12 * (1 + param.z)**(-1/2) * param.epsB**(-3/2) * (param.numDens * 1e-6)**(-1) * (param.blastEnergy * 1e7/1e52)**(-1/2) * (param.tZ * ( 1 + param.z)/86400)**(-1/2)
    elif param.k == 2: # TODO: taken from Laskar's gstable2.py discuss where it comes from
        return 2.34e10 * (1 + param.z)**(-1.5) * param.Astar**(-2.) * param.epsB**(-1.5) * (param.blastEnergy * 1e7/1e52)**(0.5) * (param.tZ * ( 1 + param.z)/86400)**(0.5)

def gsNus(param):
    """Obtain the synchrotron nuM and nuC from GS01.
    
        parameters
        ----------
        param : ConvertParam Object
            The GRB parameters as a ConvertParam Object
    """
    
    #print(params)
        
    num     = gsNuMslow(param)
    nuc     = gsNuCslow(param)
    if (nuc < num):        
        num = gsNuMfast(param)
        nuc = gsNuCfast(param)
        
    return (num, nuc)
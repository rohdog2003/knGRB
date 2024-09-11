# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:10:51 2024

@author: rohdo
"""
# Note use of SI kg m s 
import scipy.constants as const
import numpy as np
from gsNus import gsNus

class ConvertParam:
    pi = const.pi
    muNaught = const.mu_0
    mP = const.m_p
    mE = const.m_e
    qE = const.e
    c = const.c
    
    def __init__(self, Astar = 1, blastEnergy = 1e45, epsB = 0.1, epsE = 0.1, tZ = 86400, numDens = 1e6, p = 2.5, k = 0, \
                 z = 1, gsGammas = False):
        """"""
        self.numDens = numDens
        
        if k > 0:
            self.A = 5e10 * Astar 
        else:
            self.A = self.mP * self.numDens
        
        self.Astar = Astar
        self.blastEnergy = blastEnergy
        self.epsB = epsB
        self.epsE = epsE
        self.tZ = tZ
        self.p = p
        self.k = k
        self.z = z
        self.gsGammas = gsGammas
    
    def shockRad(self):
        """Shock radius. Zhang (8.30)
        """
        return (((17 - 4*self.k) * (4 - self.k) * self.blastEnergy * self.tZ)/\
                (4 * self.pi * self.A * self.c ))**(1/(4-self.k))
            
    def rho(self):
        """Wind density.
        """
        if self.k > 0:
            return self.A * (self.shockRad())**(-self.k)
        else:
            return self.mP * self.numDens
    
    def gammaBulk(self): 
        """Fluid Bulk lorentz factor. Zhang (8.31) GS01 (A15).
        """
        return np.sqrt(self.shockRad()/(4 * (4 - self.k) * self.c * self.tZ))
    
    def energyDens(self): # shock lorentz factor implemented
        """Energy Density. Granot & Sari (A4)
        """
        return  4 * self.gammaBulk()**2 * self.rho() * self.c**2
    
    def magField(self):
        """Magnetic field strength. Granot & Sari (A2)
        """
        return np.sqrt(2 * self.muNaught * self.epsB * self.energyDens())
    
    def gammaM(self): # fluid lorentz factor implemented (what lorentz factor should I be using?)
        """Injection lorentz factor. Granot & Sari (A1)
        """
        if self.gsGammas == True:
            return self.nuSynInv(gsNus(self)[0])
        else:
            return (self.p-2)/(self.p-1) * self.mP/self.mE * self.epsE * self.gammaBulk()
    
    def gammaC(self): # shock lorentz factor implemented
        """Calculates the cooling lorentz factor for the synchrotron spectrum.
        
            parameters
            ----------
            gammaBulk : float
                Bulk lorentz factor.
                
            magField : float
                Magnetic field strength.
                
            tZ : float
                Time in the frame of the observer.
        """
        if self.gsGammas == True:
            return self.nuSynInv(gsNus(self)[1])
        else:
            epsilonNaught = const.epsilon_0
            muNaught = const.mu_0
            e = const.e
            m = const.m_e
            c = const.c
            pi = const.pi
            
            classicalElectronRadius = (1/(4*pi*epsilonNaught))*(e**2/(m*c**2))
            thomsonCrossSection = (8*pi/3)*classicalElectronRadius**2
            
            t = 2 * self.gammaBulk() * self.tZ #see NAS09 (9)
            
            return (3/2) * (muNaught * m * c)/(thomsonCrossSection * self.magField()**2 * t)
    
    def neiso(self, nSwept = False):
        """The isotropic equivalent number of emitting electrons. Yamasaki (10)
        modified to include k=0 and k=2 density profiles.
        """
        fP = (self.p - 1)/(self.p - 2)
        
        if nSwept == True:
            return (4/3) * self.pi * self.rho()/self.mP * self.shockRad()**3
        else:
            return (self.pi * 4**3 * (4 - self.k)**3 * self.c * self.tZ**3)/(6 * self.muNaught * fP * self.mE) *\
                (self.epsE/self.epsB) * (self.magField()**2 * self.gammaBulk()**5)/(self.gammaM())
            
    def nSweptPerSec(self): # shock lorentz factor implemented
        """The number of electrons swept per second. Assumes gammaBulk >> 1 using
        Huang (7) for k=0 case. The expression given in Huang is a factor of 4/3
        more.
        """
        return self.gammaBulk() * 4/3 * (3 - self.k) * self.pi * self.rho()/self.mP * \
            self.shockRad()**2 * self.c
        
    def qNaught(self):
        """Injection rate at gammaM using Huang (6) for gamma = gammaM.
        """
        return self.nSweptPerSec() * (self.p - 1)/self.gammaM()
    
    def nuSynInv(self, nu):
        """The comovin Lorentz factor that produces the most synchrotron radiation at a 
        given frequency in the observer frame.
        """
        nuG = self.qE * self.magField()/(2 * self.pi * self.mE)
        gamma = np.sqrt((1 + self.z) * nu/(self.gammaBulk() * nuG))
            
        return gamma
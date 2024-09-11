# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:03:27 2024

@author: rohdo
"""

import numpy as np
from SynchrotronFunction import synF
from SynchrotronSelfComptonFunction import gKN
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.optimize import fsolve
import scipy.constants as const
from cosmoCalc import lumDistLCDM
import comptonYnorm

def gammaC(gammaBulk, magField, tZ): # shock lorentz factor implemented
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
    epsilonNaught = const.epsilon_0
    muNaught = const.mu_0
    e = const.e
    m = const.m_e
    c = const.c
    pi = const.pi
    
    classicalElectronRadius = (1/(4*pi*epsilonNaught))*(e**2/(m*c**2))
    thomsonCrossSection = (8*pi/3)*classicalElectronRadius**2
    
    t = gammaBulk * tZ #comoving time
    
    return (3/2) * (muNaught * m * c)/(thomsonCrossSection * magField**2 * t)

class Afterglow:
    
    def __init__(self, param, step = True, fKN = 1, gsGammaCsyn = -1,\
                     Ynorm = "reg", Fnorm = "reg"):
        """Attributes of an afterglow including cooling critical lorentz factor.
        
            parameters
            ----------
            param: ConvertParam object
                The parameters of the GRB.
                
            step: boolean (default = True)
                Whether or not to apply the step function approximation.
            
            fKN: float
                Where to cut off the step function approximation.
                
            gsGammaCsyn: float
                first estimate of the cooling Lorentz factor.
        """
        
        self.Ynorm = Ynorm
        self.Fnorm = Fnorm
        
        epsilonNaught = const.epsilon_0
        e = const.e
        m = const.m_e
        c = const.c
        pi = const.pi
        
        classicalElectronRadius = (1/(4*pi*epsilonNaught))*(e**2/(m*c**2))
        self.thomsonCrossSection = (8*pi/3)*classicalElectronRadius**2
        
        self.fKN = fKN
        self.gammaBulk = param.gammaBulk()
        self.magField = param.magField()
        self.tZ = param.tZ
        self.p = param.p
        self.z = param.z
        self.k = param.k
        self.step = step
        
        self.epsB = param.epsB
        self.epsE = param.epsE
        self.neiso = param.neiso()
        self.qNaught = param.qNaught()
        
        self.gammaM = param.gammaM()
        
        if gsGammaCsyn != -1:
            self.gammaCsyn = gsGammaCsyn
            self.gammaC = gsGammaCsyn
        else:
            self.gammaCsyn = gammaC(param.gammaBulk(), param.magField(), param.tZ)
            self.gammaC = gammaC(param.gammaBulk(), param.magField(), param.tZ)
        
        #self.NNaught = self.gammaBulk * self.neiso # total number of electrons in injection distribution
        self.elecSpecNorm = self.elecSpecTemplateNorm() # FIX electron dist integral is gammaBulk * Nswept
        #self.elecSpecNorm = (self.p - 1) * self.NNaught * self.gammaM**(self.p - 1)
        self.gammaHatM = self.gammaHat(self.gammaM)
        self.gammaHatC = self.gammaHat(self.gammaC)
        
        #self.thomsonY = self.getThomsonY(self.elecSpecTemplate) # Incorrect cuz Jacovitch has errors and slow to fast cooling time unknown
        
        self.gammaN = max(self.gammaHatM, self.gammaHatC)

        self.synSpecInterp = self.synSpecUnnormalizedInterp(self.elecSpecTemplate)
        self.comptonYInterp = self.comptonYNormalizedInterp()
        
        self.Yc = self.comptonY(self.gammaC)
        
        self.lumDist = lumDistLCDM(param.z) # assumes LCDM cosmology
        
        uSubFactor = (3/4) * (2/3)**(3/2) * self.nuG()**(-1/2) 
        pitchAngleIntFactor = 1/2
        emissivityFactor = np.sqrt(3) * const.e**3 * self.magField/(2 * np.pi * const.epsilon_0 * const.c * const.m_e) # look at factor of Longair (8.58)
        self.synSpecNormFactor = (1+self.z)/(4*np.pi*self.lumDist**2) * uSubFactor * pitchAngleIntFactor * emissivityFactor * self.elecSpecNorm * self.gammaBulk
        
        self.observerSynSpecInterp = ([nu * self.gammaBulk/(1 + self.z) for nu in self.synSpecInterp[0]], [F * self.synSpecNormFactor for F in self.synSpecInterp[1]])
        
    def nuSyn(self, gamma, redshift = True):
        """Characteristic synchrotron frequency as a function of the bulk lorentz 
        factor, magnetic field, and electron lorentz factor.
        
            parameters
            ----------
            gamma : float
                Electron lorentz factor.
        """
        e = const.e
        m = const.m_e
        #c = const.c
        pi = const.pi
        
        nu = (self.gammaBulk * e * self.magField * gamma**2) / (2 * pi * m)
        
        if redshift == True:
            nu /= 1 + self.z
            
        return nu
    
    def nuSynInv(self, nu):
        """The comovin Lorentz factor that produces the most synchrotron radiation at a 
        given frequency in the observer frame.
        """
        nuG = const.e * self.magField/(2 * const.pi * const.m_e)
        gamma = np.sqrt((1 + self.z) * nu/(self.gammaBulk * nuG))
            
        return gamma
    
    def nuSynPrime(self, gamma):
        """Characteristic synchrotron frequency in the comoving frame as in
        Granot and Sari (A17).
        
            parameters
            ----------
            gamma : float
                Electron lorentz factor.
        """
        e = const.e
        m = const.m_e
        #c = const.c
        pi = const.pi
        
        return (e * self.magField * gamma**2) / (2 * pi * m)
    
    def nuG(self):
        """ comoving gyrofrequency.
        """
        e = const.e
        m = const.m_e
        #c = const.c
        pi = const.pi
        
        return (e * self.magField)/(2 * pi * m) # gyrofrequency Longair 8.55
    
    def elecSpecTemplateNorm(self):
        """Found by inserting the proportional expressions for the electron 
        distribution given in NAS09 into the continuity equation
        """
        t = self.gammaBulk * self.tZ # comoving time REMOVED factor of 2
        
        if self.gammaC < self.gammaM: # fast cooling
            return self.gammaM * self.gammaCsyn * self.qNaught * t/(self.p - 1)
            #return self.neiso/(self.gammaC**(-1) - ((self.p - 1)/self.p) * self.gammaM**(-1)) # removed factor of gammaBulk
        elif self.gammaM < self.gammaC: # slow cooling
            return self.gammaM**self.p * self.qNaught * t/(self.p - 1)
            #return (self.p - 1) * self.neiso/(self.gammaM**(-self.p + 1) - (1/self.p) * self.gammaC**(-self.p + 1)) # removed factor of gammaBulk
        
    
    def elecSpecTemplate(self, gamma):
        """Electron spectrum/distribution dN/dγ template function ignoring KN 
        effects. Returns dN/dγ for a value of gamma under conditions of cooling 
        factor gammaC, injection factor gammaM, and power law exponent p.
        
            parameters
            ----------
            gamma : float
                Electron lorentz factor.
        """
        if self.gammaCsyn < self.gammaM: # Fast cooling.
            if self.gammaCsyn <= gamma <= self.gammaM:
                return gamma**(-2)
            elif self.gammaM <= gamma:
                return self.gammaM**(self.p-1) * gamma**(-self.p - 1)
            elif gamma < self.gammaCsyn:
                return 0
        elif self.gammaM < self.gammaCsyn: # Slow cooling.
            if self.gammaM <= gamma <= self.gammaCsyn:
                return gamma**(-self.p)
            elif self.gammaCsyn <= gamma:
                return self.gammaCsyn * gamma**(-self.p - 1)
            elif gamma < self.gammaM:
                return 0
    
    def synSpecTemplateUnnormalized(self, nu):
        """The template electron distribution from the integral of the 
        emissivity of a single electron and the electron distribution 
        unnormalized.
        
            parameters
            ----------
            nu : float
                Frequency.
        """
        
        return self.synSpecUnnormalized(self.elecSpecTemplate, nu)
    
    def synSpecTemplate(self, nu):
        """The normalized template electron distribution from the integral of 
        the emissivity of a single electron and the electron distribution.
        
            parameters
            ----------
            neiso : float
                The isotropic equivalent number of emitting electrons.
            
            nu : float
                Frequency.
        """
        return self.synSpec(self.elecSpecTemplate, nu)
    
    def powerSyn(self, gamma): #check conversion 
        """Synchrotron power as defined in Yamasaki (9)
        """
        muNaught = const.mu_0
        c = const.c
        
        return (2/(3 * muNaught)) * self.thomsonCrossSection * c * self.magField**2 *\
            gamma**2 * self.gammaBulk**2
    
    def yamasakiFastNorm(self):
        """Yamasaki normalization for fast cooling (8).
        """
        nuC = self.nuSyn(self.gammaC, redshift = True) # see Yamasaki (1)
        
        return (1+self.z)/(4*np.pi*self.lumDist**2) * (self.neiso * self.gammaBulk * self.gammaC * const.m_e * const.c**2/\
                (nuC * (1 + self.Yc) * self.tZ * (1 + self.z)))
        
    def yamasakiSlowNorm(self):
        """Yamasaki normalization for slow cooling (9).
        """
        nuM = self.nuSyn(self.gammaM, redshift = True)
        
        return (1+self.z)/(4*np.pi*self.lumDist**2) * (self.neiso * self.powerSyn(self.gammaM))/nuM
    
    def normFactor(self, elecSpec): # TODO: Yamasaki does not work well for numerically smoothed.
        """The normalization factor for the synchrotron spectrum.
        """
        nuM = self.nuSyn(self.gammaM, redshift = True)
        nuC = self.nuSyn(self.gammaC, redshift = True)
        
        nuM *= (1 + self.z)/(self.gammaBulk)
        nuC *= (1 + self.z)/(self.gammaBulk)
        
        if self.gammaM < self.gammaC: #slow cooling
            normFactor = self.yamasakiSlowNorm() / self.synSpecUnnormalized(elecSpec, nuM)
        elif self.gammaC < self.gammaM: #fast cooling
            normFactor = self.yamasakiFastNorm() / self.synSpecUnnormalized(elecSpec, nuC)
            
        return normFactor 
    
    def synSpec(self, elecSpec, nu):
        """Normalized synchrotron spectrum redshifted to the observer frame.
            
            parameters
            ----------
            elecSpec : callable
                The electron distribution as a function of gamma.
                
            nu : float
                Frequency.
        """
        if self.Fnorm == "YP":
            normFactor = self.normFactor(elecSpec)
        elif self.Fnorm == "reg":
            uSubFactor = (3/4) * (2/3)**(3/2) * self.nuG()**(-1/2) 
            pitchAngleIntFactor = 1/2
            emissivityFactor = np.sqrt(3) * const.e**3 * self.magField/(2 * np.pi * const.epsilon_0 * const.c * const.m_e) # look at factor of Longair (8.58)
            normFactor = (1+self.z)/(4*np.pi*self.lumDist**2) * uSubFactor * pitchAngleIntFactor * emissivityFactor * self.elecSpecNorm * self.gammaBulk # TODO: normalization does not work for wind case
            
        # doppler should be a factor of 2 * gammaBulk but removing factor of 2
        # seems to work better.
        nu *= (1 + self.z)/(self.gammaBulk) #* (1/2)
        
        return normFactor * self.synSpecUnnormalized(elecSpec, nu)
    
    def synSpecUnnormalized(self, elecSpec, nu):
        """The unnormalized synchrotron spectrum in the comoving frame.
        
            parameters
            ----------
            elecSpec : callable
                The electron distribution as a function of gamma.
            
            nu : float
                Frequency.
        """
        nuG = self.nuG()
        nuM = self.nuSynPrime(self.gammaM)
        nuC = self.nuSynPrime(self.gammaC)
        
        fudgeFactor = 0.1 # Not sure if should be parameter dependent.
        
        def elecSpecU(u, alpha):
            """"""
            gamma = np.sqrt(nu/((3/2) * u * nuG * np.sin(alpha)))
                            
            return elecSpec(gamma)
        
        def integrand(u, alpha):
            """"""
            return np.sin(alpha)**(3/2) * u**(-3/2) * synF(u) * elecSpecU(u, alpha)
        
        def hfun(alpha):
            """"""
            gammaLow = min(self.gammaM, self.gammaC)
            return nu/(3/2 * gammaLow**2 * self.nuG() * np.sin(alpha))
        
        fudgeNu = fudgeFactor * min(nuM, nuC)
        
        if nu < fudgeNu: # dblquad breaks down in this regime.
            fudgeNu = fudgeFactor * min(nuM, nuC)
            return nu**(1/3) * self.synSpecUnnormalized(elecSpec, fudgeNu)/fudgeNu**(1/3)
        elif nu < min(nuM, nuC) * 1/fudgeFactor:
            return np.sqrt(nu) * dblquad(integrand, 0, np.pi, 0, hfun)[0]
        else:
            return np.sqrt(nu) * dblquad(integrand, 0, np.pi, 0, np.inf)[0]
    
    def gammaNaught(self):
        """McCarthy and Laskar (26)
        """
        return fsolve(lambda gamma : np.interp(gamma, self.comptonYInterp[0], \
                                               self.comptonYInterp[1]) - 1, \
                      min(self.gammaHatM, self.gammaHatC))[0]
    
    def gammaSelf(self):
        """McCarthy and Laskar (13)
        """
        Bqed = 4.41e9 # need greater precision
        return ( self.fKN * Bqed/self.magField)**(1/3)
    
    def gammaHat(self, gamma):
        """NAS09 (4)
        """
        return self.gammaSelf()**3/gamma**2
    
    def gammaTilde(self, gamma):
        """NAS09 (5) subsituting (4)
        """
        return (gamma * self.gammaHat(gamma))**(1/2)
    
    def nuSynNaught(self, redshift = True):
        """In the observer frame.
        """
        return self.nuSyn(self.gammaNaught(), redshift = redshift)
    
    def nuSynTilde(self, gamma, redshift = True):
        """In the observer frame.
        """
        return self.nuSyn(self.gammaTilde(gamma), redshift = redshift)
    
    def nuSynTildePrime(self, gamma):
        """In the comoving frame.
        """
        return self.nuSynPrime(self.gammaTilde(gamma))
    
    def nuSynHat(self, gamma, redshift = True):
        """In the observer frame.
        """
        return self.nuSyn(self.gammaHat(gamma), redshift = redshift)
    
    def nuICm(self, redshift = True):
        """"""
        return 2 * self.gammaM**2 * self.nuSyn(self.gammaM, redshift = True)
        
    def nuICc(self, redshift = True):
        """"""
        return 2 * self.gammaC**2 * self.nuSyn(self.gammaC, redshift = True)
        
    def nuICkn(self, redshift = True):
        """"""
        return self.gammaC**2 * self.nuSynTilde(self.gammaC, redshift = True)
        
    def nuIChatC(self, redshift = True):
        """"""
        return self.gammaHatC**2 * self.nuSyn(self.gammaC, redshift = True)
        
    def synSpecUnnormalizedInterp(self, elecSpec, res = 20, tol = 7):
        """Constructs an interpolation of the unnormalized comoving synchrotron 
        spectrum.
        """
        nuBreaks = [self.nuSynPrime(self.gammaM), self.nuSynPrime(self.gammaC),\
                    self.nuSynPrime(self.gammaHatM), self.nuSynPrime(self.gammaHatC),\
                        self.nuSynPrime(self.gammaN)]
        nuBreaks.sort()
            
        nuInterp = np.array([])
            
        nuInterp = np.append(nuInterp, np.geomspace(10**(-tol) * nuBreaks[0], nuBreaks[0], num = res))
        nuInterp = np.append(nuInterp, np.geomspace(nuBreaks[0], nuBreaks[1], num = res))
        nuInterp = np.append(nuInterp, np.geomspace(nuBreaks[1], nuBreaks[2], num = res))
        nuInterp = np.append(nuInterp, np.geomspace(nuBreaks[2], nuBreaks[3], num = res))
        nuInterp = np.append(nuInterp, np.geomspace(nuBreaks[3], nuBreaks[4], num = res))
        nuInterp = np.append(nuInterp, np.geomspace(nuBreaks[4], 10**(tol) * nuBreaks[4], num = res))
            
        nuInterp = np.unique(nuInterp)
            
        synSpecUnnormalizedInterp = np.array([self.synSpecUnnormalized(elecSpec, nu) for nu in nuInterp])
        
        return (nuInterp, synSpecUnnormalizedInterp)
    
    def knCrossSection(x):
        """The Klein-Nishina QFT full cross section as a function of
        x = hbar * angularFrequency/electronRestEnergy.
        """
        epsilonNaught = const.epsilon_0
        e = const.e
        m = const.m_e
        c = const.c
        pi = const.pi
        
        x = x/(2 * pi)
        
        classicalElectronRadius = (1/(4*pi*epsilonNaught))*(e**2/(m*c**2))
        
        return pi * classicalElectronRadius**2 * 1/x * \
            ((1 - 2 * (x+1)/x**2) * np.log(2 * x + 1) + 0.5 + 4/x - 1/(2 * (2 * x + 1)**2))
    
    def knIntegral(self, x):
        """Approximately the inner integral of NAS09 (14) as a function of 
        x = nu/nuSynTilde.
        """
        if x <= 2.9013:
            return 1/(3.62 * x + 1)
        else:
            return (2 * np.log(x) - 1 + np.log(16))/x**2
    
    def comptonYNormalizedInterp(self, res = 20, tol = 7):
        """"""
        gammaBreaks = [self.gammaM, self.gammaC, self.gammaHatM, self.gammaHatC, self.gammaN]
        gammaBreaks.sort()
            
        gammaInterp = np.array([])
            
        gammaInterp = np.append(gammaInterp, np.geomspace(1, gammaBreaks[0], num = res))
        gammaInterp = np.append(gammaInterp, np.geomspace(gammaBreaks[0], gammaBreaks[1], num = res))
        gammaInterp = np.append(gammaInterp, np.geomspace(gammaBreaks[1], gammaBreaks[2], num = res))
        gammaInterp = np.append(gammaInterp, np.geomspace(gammaBreaks[2], gammaBreaks[3], num = res))
        gammaInterp = np.append(gammaInterp, np.geomspace(gammaBreaks[3], gammaBreaks[4], num = res))
        gammaInterp = np.append(gammaInterp, np.geomspace(gammaBreaks[4], 10**(tol) * gammaBreaks[4], num = res))
            
        gammaInterp = np.unique(gammaInterp)
            
        comptonYNormalizedInterp = np.array([self.comptonY(gamma) for gamma in gammaInterp])
        
        return (gammaInterp, comptonYNormalizedInterp)
    
    def comptonY(self, gamma, res = 20, tol = 7, msg = False, template = False,\
                 unnormalized = False):
        """Calculates the comptonY parameter at a certain gamma.
        
            parameters
            ----------
            elecSpec : callable
                The electron distribution function.
            
            gamma : float
                The lorentz factor of the electron.
        """
        if unnormalized == False:
            if self.Ynorm == "reg":
                uSubFactor = (3/4) * (2/3)**(3/2) * self.nuG()**(-1/2) 
                pitchAngleIntFactor = 1/2
                emissivityFactor = np.sqrt(3) * const.e**3 * self.magField/(2 * np.pi * const.epsilon_0 * const.c * const.m_e) # look at factor of Longair (8.58)
                energyInjectionRate = self.qNaught * const.m_e * const.c**2 * self.gammaM**2/(self.p - 2)
                
                normFactor = (self.epsE/self.epsB) * (1/energyInjectionRate) * uSubFactor * emissivityFactor * pitchAngleIntFactor * self.elecSpecNorm #* fudgeFactor
        
            elif self.Ynorm == "JBH":
                normFactor = comptonYnorm.JBHcomptonYnormFactor(self, self.comptonY, kwargs = {"unnormalized" : True})
            elif self.Ynorm == "SE":
                normFactor = comptonYnorm.SEcomptonYnormFactor(self, self.comptonY, kwargs = {"unnormalized" : True})
            elif self.Ynorm == "NAS":
                normFactor = comptonYnorm.NAScomptonYnormFactor(self, self.comptonY, kwargs = {"unnormalized" : True}) # fails to converge for case1_2
            
        else:
            normFactor = 1
        
        nuInterp, synSpecUnnormalizedInterp = self.synSpecInterp
        
        if msg:
            print("Interpolation constructed")
        
        def integrand(nu):
            """The interpolated unnormalized synchrotron spectrum.
            """
            return np.interp(nu, nuInterp, synSpecUnnormalizedInterp) 
        
        def fullKNIntegrand(nu):
            """"""
            return self.knIntegral(nu/self.nuSynTildePrime(gamma)) * integrand(nu)
        
        points = [self.nuSynPrime(self.gammaM),\
                  self.nuSynPrime(self.gammaC),\
                      self.nuSynPrime(self.gammaHatM),\
                          self.nuSynPrime(self.gammaHatC)]
        
        if self.step == True:
            sol = quad(integrand, 0, self.fKN * self.nuSynTildePrime(gamma), points = points)
            if msg == True:
                print("error: {}".format(normFactor * sol[1]))
            return normFactor * sol[0]
        else:
            sol = quad(fullKNIntegrand, 0, 100 * self.nuSynTildePrime(gamma), points = points)
            if msg == True:
                print("error: {}".format(normFactor * sol[1]))
            return normFactor * sol[0]
        
    def elecSpecCompton(self, gamma):
        """Electron spectrum/distribution dN/dγ template function including KN
        effects in an iterative way. Returns dN/dγ for a value of gamma under 
        conditions of cooling factor gammaC, injection factor gammaM, 
        and power law exponent p.
        
            parameters
            ----------
            gamma : float
                Electron lorentz factor.
        """
        gammaInterp, comptonYInterp = self.comptonYInterp
        
        # See equations (12) and (13) of NAS09.
        if self.gammaC < self.gammaM: # Fast cooling.
            if self.gammaC <= gamma <= self.gammaM:
                return gamma**(-2)/(1 + np.interp(gamma, gammaInterp, comptonYInterp))
            elif self.gammaM <= gamma:
                return self.gammaM**(self.p-1) * gamma**(-self.p - 1)/(1 + np.interp(gamma, gammaInterp, comptonYInterp))
            elif gamma < self.gammaC:
                return 0
        elif self.gammaM < self.gammaC: # Slow cooling.
            if self.gammaM <= gamma <= self.gammaC:
                return gamma**(-self.p)
            elif self.gammaC <= gamma:
                return self.gammaCsyn * gamma**(-self.p - 1)/(1 + np.interp(gamma, gammaInterp, comptonYInterp))
            elif gamma < self.gammaM:
                return 0
    
    
    def synSelfComptonSpec(self, nuIC, epsabs = 0.05, epsrel = 0.05):
        """The synchtrotron self compton spectrum as in Lemoine (17).
        """
        nuSynTildeMax = self.nuSynTilde(1) # maximum value of nuSynTilde in observer frame.
        shockRad = 4 * (4 - self.k) * const.c * self.tZ * self.gammaBulk**2
        normFactor = 3 * self.thomsonCrossSection/(4 * np.pi * shockRad**2) * 1/nuSynTildeMax
        lowerBoundW = max(0, 1 - nuIC/nuSynTildeMax * 1/min(self.gammaM, self.gammaC))
        
        def elecSpecW(w):
            return self.elecSpecNorm * self.elecSpecCompton(nuIC/nuSynTildeMax * 1/(1 - w))
        
        def synSpecWQ(w, q):
            nu = nuSynTildeMax**2/nuIC * (1 - w)**2/(4 * q * w)
            return np.interp(nu, self.observerSynSpecInterp[0], self.observerSynSpecInterp[1])
        
        def integrand(w, q):
            if w == 0 or w == 1: # avoid float division by zero
                return 0
            else:
                return w/(1 - w)**2 * elecSpecW(w) * gKN(q, w) * synSpecWQ(w, q)
        
        result = normFactor * nuIC * dblquad(integrand, 0, 1, lowerBoundW, 1, epsabs = epsabs, epsrel = epsrel)[0]
        #print(result)
        return result
    
    def iterate(self):
        """Gets the next estimate of the synchrotron and SSC cooled electron
        distribution."""
        self.gammaC = self.gammaCsyn/(1 + self.Yc)
        self.elecSpecNorm = self.elecSpecTemplateNorm()
        uSubFactor = (3/4) * (2/3)**(3/2) * self.nuG()**(-1/2) 
        pitchAngleIntFactor = 1/2
        emissivityFactor = np.sqrt(3) * const.e**3 * self.magField/(2 * np.pi * const.epsilon_0 * const.c * const.m_e) # look at factor of Longair (8.58)
        self.synSpecNormFactor = (1+self.z)/(4*np.pi*self.lumDist**2) * uSubFactor * pitchAngleIntFactor * emissivityFactor * self.elecSpecNorm * self.gammaBulk
        self.gammaHatC = self.gammaHat(self.gammaC)
        self.synSpecInterp = self.synSpecUnnormalizedInterp(self.elecSpecCompton)
        self.comptonYInterp = self.comptonYNormalizedInterp()
        self.Yc = self.comptonY(self.gammaC)
        self.gammaN = self.gammaNaught()
        self.observerSynSpecInterp = ([nu * self.gammaBulk/(1 + self.z) for nu in self.synSpecInterp[0]], [F * self.synSpecNormFactor for F in self.synSpecInterp[1]])
        
        
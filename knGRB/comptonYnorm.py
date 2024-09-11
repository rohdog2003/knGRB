# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:43:06 2024

@author: rohdo
"""
import numpy as np

def SEcomptonYnormFactor(glow, unnormalizedComptonY, kwargs = {}):
    """Normalizations for the Compton Y parameter as given in SE01 (3.2).
    
        parameters
        ----------
        glow : object
            The Afterglow object containing all the afterglow information.
            
        unnormalizedComptonY : callable
            The function that calculates the shape of the Compton Y which is not
            necessarily normalized.
    """
    eta = (glow.gammaC/glow.gammaM)**(2 - glow.p)
    YT = (-1 + np.sqrt(1 + 4 * eta * glow.epsE/glow.epsB))/2
    return YT/unnormalizedComptonY(1, **kwargs)

def NAScomptonYnormFactor(glow, unnormalizedComptonY, kwargs = {}):
    """Normalizations for the Compton Y parameter as given in NAS09 (26), (47)
    and section 3.1.
    
        parameters
        ----------
        glow : object
            The Afterglow object containing all the afterglow information.
            
        unnormalizedComptonY : callable
            The function that calculates the shape of the Compton Y which is not
            necessarily normalized.
    """
    if glow.gammaM < glow.gammaC: # slow cooling
        # solving NAS09 (47) Yc*(1+Yc)=RHS
        eta = (glow.gammaC/glow.gammaM)**(2 - glow.p)
        RHS = glow.epsE/glow.epsB * eta * (min(glow.gammaC, glow.gammaHatC))**((3 - glow.p)/2)
        Yc = (-1 + np.sqrt(1 + 4 * RHS))/2
        return Yc/unnormalizedComptonY(glow.gammaC, **kwargs)
    else: # fast cooling
        if glow.gammaM <= glow.gammaHatM: # weak KN and NAS09 case III
            Ym = np.sqrt(glow.epsE/glow.epsB)
            return Ym/unnormalizedComptonY(glow.gammaM, **kwargs)
        else: # strong KN
            # solving NAS09 (26) Yself*(1 + Yself) = RHS
            RHS = glow.epsE/glow.epsB * glow.gammaSelf()/glow.gammaM # consider changing glow.gammaSelf() from a function to a self variable if too slow
            Yself = (-1 + np.sqrt(1 + 4 * RHS))/2
            return Yself/unnormalizedComptonY(glow.gammaSelf(), **kwargs)

def JBHcomptonYnormFactor(glow, unnormalizedComptonY, kwargs = {}):
    """Normalizations for the Compton Y parameter to the Thomson Y as in JBH21 
    with smoothing as in McCarthy and Laskar.
    
        parameters
        ----------
        glow : object
            The Afterglow object containing all the afterglow information.
            
        unnormalizedComptonY : callable
            The function that calculates the shape of the Compton Y which is not
            necessarily normalized.
    """
    return yt(glow)/unnormalizedComptonY(1, **kwargs)    
    

def yt(glow):

    p = glow.p

    # Alpha as seen in JBH Eq.13 for smoothing.
    a = -60 * p ** -2

    YT = (
        YT_fast(glow) ** a + YT_slow(glow) ** a
    ) ** (1 / a)

    return YT

# Solves A7 by passing coeffs of A7 to cubic_formula()
def YT_fast(glow):
    p = glow.p
    gammam = glow.gammaM
    gammacs = glow.gammaCsyn
    E_ratio = glow.epsE / glow.epsB
    gammacsover_m = gammacs / gammam + 0j
    a = 1
    b = 2 - (p - 1) / p * gammacsover_m
    c = 1 - E_ratio - (p - 1) / p * gammacsover_m
    d = E_ratio * ((p - 2) / (p - 1) * gammacsover_m - 1)
    return cubic_formula(glow, a, b, c, d)


# Cubic formula from applying Cardano's method to a general cubic of
# coefficients ax**3 + bx**2 + cx + d = 0.
def cubic_formula(glow, a, b, c, d):
    solution = 0 + 0j
    A = -(b ** 3) / (27 * a ** 3) + b * c / (6 * a ** 2) - d / (2 * a)
    B = c / (3 * a) - b ** 2 / (9 * a ** 2)
    solution = (
        (A + (A ** 2 + B ** 3) ** (1 / 2)) ** (1 / 3)
        + (A - (A ** 2 + B ** 3) ** (1 / 2)) ** (1 / 3)
        - b / (3 * a)
    )
    return solution.real


# Computes Y Thomson in the slow regime by smoothing between the approximations
# in JBH Tab.2.
def YT_slow(glow):
    # FIXME YMMV with this smoothing constant.
    # Works well for JBH Fig.1 & Fig.1 parameters.
    a = -1.7
    return (
        YT_slow_approx(glow, 2) ** a
        + YT_slow_approx(glow, 3) ** a
    ) ** (1 / a)


# Returns an approximation for Y_slow as given in table 2 of JBH.
# t_2_row is the row number of the approximation in the table.
def YT_slow_approx(glow, t_2_row):
    p = glow.p
    E_ratio = glow.epsE / glow.epsB
    gammam = glow.gammaM
    gammacs = glow.gammaCsyn
    inner_term = E_ratio / (3 - p) * (gammam / gammacs) ** (p - 2)
    # print(inner_term)
    # Analytic solution gives approximation for large Y.
    if t_2_row == 2:
        # print(1/(4-p))
        # print(inner_term ** (1 / (4 - p)))
        return inner_term ** (1 / (4 - p))
    # Analytic solution gives approximation for small Y.
    elif t_2_row == 3:
        return inner_term
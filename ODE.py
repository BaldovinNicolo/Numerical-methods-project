#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:47:46 2022

@author: nicolobaldovin
"""

import numpy as np
import matplotlib.pyplot as plt




# Define the sigmoid function
def sigmoid(u):
    '''
    
    Parameters
    ----------
    u : 
        TYPE: 
            - np.ndarray or float
        DESCRIPTION:
            - input value or array of values to apply the sigmoid function to


    Returns
    -------
    result :
        TYPE:
            - np.ndarray or float
        DESCRIPTION:
            - result of applying the sigmoid function
            
    '''
    
    return 1 / (1 + np.exp(-u))




# Implement the Euler Forward method for solving differential equations
def Eulero_Forward(func, points, X):
    '''
    
    Parameters
    ----------
    func : 
        TYPE: 
            - function
        DESCRIPTION:
            - the function representing the differential equation u' = func(t, u)
    
    points : 
        TYPE: 
            - np.ndarray
        DESCRIPTION:
            - array of time points at which to solve the differential equation
    
    X : 
        TYPE: 
            - float
        DESCRIPTION:
            - initial condition for the differential equation
    
    Returns
    -------
    u : 
        TYPE:
            - np.ndarray
        DESCRIPTION:
            - array of solution values at the specified time points
            
    '''
    
    n = len(points)
    u = np.zeros(n)     
    u[0] = X           # Initial condition
    for j in range(n-1):
        h = points[j+1] - points[j]
        u[j+1] = u[j] + (h * func(points[j], u[j]))  # Euler Forward update
    return u



# Implement the secant method for finding roots of a function
def secanti(x0, x1, funz, e = 1.0e-6, max_iter = 100):
    '''
    
    Parameters
    ----------
    x0 : 
        TYPE: 
            - float
        DESCRIPTION:
            - initial guess for the root
    
    x1 : 
        TYPE: 
            - float
        DESCRIPTION:
            - second guess for the root
    
    funz : 
        TYPE: 
            - function
        DESCRIPTION:
            - function for which the root is to be found
    
    e : 
        TYPE: 
            - float, optional
        DESCRIPTION:
            - tolerance for the root finding algorithm, default is 1.0e-6
    
    max_iter : 
        TYPE: 
            - int, optional
        DESCRIPTION:
            - maximum number of iterations, default is 100
    
    Returns
    -------
    x1 : 
        TYPE:
            - float
        DESCRIPTION:
            - approximation to the root
            
    '''
    
    iter = 0
    while abs(x1 -x0) > e and iter < max_iter:
        iter += 1
        x1 = x1 - (funz(x1) * (x1 - x0) / (funz(x1) - funz(x0)))    # Secant method update
        x0 = x1
    return x1



# Implement the Euler Backward method for solving differential equations
def Eulero_Backward(func, points, X):
    '''
    Parameters
    ----------
    func : 
        TYPE: 
            - function
        DESCRIPTION:
            - the function representing the differential equation u' = func(t, u)
    
    points : 
        TYPE: 
            - np.ndarray
        DESCRIPTION:
            - array of time points at which to solve the differential equation
    
    X : 
        TYPE: 
            - float
        DESCRIPTION:
            - initial condition for the differential equation
    
    Returns
    -------
    u : 
        TYPE:
            - np.ndarray
        DESCRIPTION:
            - array of solution values at the specified time points
    '''
    
    n = len(points)
    u = np.zeros(n)
    u[0] = X # Initial condition
    h1 = points[1] - points[0]
    u[1] = u[0] + (h1 * func(points[0], u[0])) # Initial step using Euler Forward
    for j in range(n-2):
        h = points[j+1] - points[j]
        # Define the function for the secant methods
        gf = lambda x: u[j+1] + h * func(points[j+2], x) - x
        u[j+2] = secanti(u[j], u[j+1], gf) # Use the secant method to solve for the next step
    return u





# Implement the Crank-Nicholson method for solving differential equations
def Crank_Nicholson(func, points, X):
    '''
    Parameters
    ----------
    func : 
        TYPE: 
            - function
        DESCRIPTION:
            - the function representing the differential equation u' = func(t, u)
    
    points : 
        TYPE: 
            - np.ndarray
        DESCRIPTION:
            - array of time points at which to solve the differential equation
    
    X : 
        TYPE: 
            - float
        DESCRIPTION:
            - initial condition for the differential equation
    
    Returns
    -------
    u : 
        TYPE:
            - np.ndarray
        DESCRIPTION:
            - array of solution values at the specified time points
    '''
    
    n = len(points)
    u = np.zeros(n)
    u[0] = X # Initial condition
    h1 = points[1] - points[0]
    u[1] = u[0] + (h1 * func(points[0], u[0])) # Initial step using Euler Forward
    for j in range(n-2):
        h = points[j+1] - points[j]
        # Define the function for the secant method with the Crank-Nicholson update
        gf = lambda x: u[j+1] - x + h * 0.5 * (func(points[j+2], x) + func(points[j+1], u[j+1]))
        u[j+2] = secanti(u[j], u[j+1], gf) # Use the secant method to solve for the next step
    return u


# Main block to execute the methods and plot the results
if __name__ == '__main__':
    end = 3
    n = 80
    X = 1
    func = lambda t,u:  -u # Define the differential equation u' = -u
    points = np.linspace(start = 0,  stop = end,  num = n)
    
    # Compute the solutions using different methods
    EF = Eulero_Forward(func, points, X)
    EB = Eulero_Backward(func, points, X)
    CN = Crank_Nicholson(func, points, X)
    
    # Plot
    fig = plt.figure(dpi = 200)
    exact_function = plt.plot(points, np.exp(-points), color = "yellow") # Plot the exact solution
    forwrd = plt.scatter(points, EF, s=1, label = 'Eulero Forward') # Plot Euler Forward solution
    backwrd = plt.scatter(points, EB, s=1, label = 'Eulero Backward') # Plot Euler Backward solution
    cranknich= plt.scatter(points, CN, s=1, label = 'Crank Nicholson') # Plot Crank-Nicholson solution
    plt.legend()
    plt.show()
    

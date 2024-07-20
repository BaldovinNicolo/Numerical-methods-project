#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 11:43:59 2022

@author: nicolobaldovin
"""

import numpy as np
import matplotlib.pyplot as plt


# NUMERICAL SCHEMES
def Euler_Forward(func, points, X):
    ''' 
    
    Explicit Euler method for numerical integration.
    
    Input:
        - func: function that represents the system of ODEs
        - points: vector of time points
        - X: initial state
    
    Output:
        - x: matrix of states at each time point
        
    '''
    n = len(points)
    x = np.zeros((n,) + X.shape) 
    x[0] = X
    for i in range(n-1):
        h = points[i+1] - points[i]
        x[i+1] = x[i] + h * func(points[i], x[i])
    return x

def Euler_Backward(func, points, X, e=1.0e-6):
    ''' 
    
    Implicit Euler method for numerical integration.
    
    Input:
        - func: function that represents the system of ODEs
        - points: vector of time points
        - X: initial state
        - e: tolerance for convergence
    
    Output:
        - x: matrix of states at each time point
        
    '''
    n = len(points)
    x = np.zeros((n,) + X.shape)
    x[0] = X 
    for i in range(n-1):
        h = points[i+1] - points[i]
        x[i+1] = x[i] + h * func(points[i], x[i])
        x_succ = x[i] + h * func(points[i+1], x[i+1])
        while np.sum(abs(x_succ - x[i+1])) > e:
            x[i+1] = x_succ
            x_succ = x[i] + h * func(points[i+1], x[i+1])
        x[i+1] = x_succ
    return x

def Crank_Nicholson(func, points, X, e=1.0e-6):
    ''' 
    
    Crank-Nicholson method for numerical integration.
    
    Input:
        - func: function that represents the system of ODEs
        - points: vector of time points
        - X: initial state
        - e: tolerance for convergence
    
    Output:
        - x: matrix of states at each time point
        
    '''
    n = len(points)
    x = np.zeros((n,) + X.shape)
    x[0] = X
    for i in range(n-1):
        h = points[i+1] - points[i]
        x[i+1] = x[i] + h * func(points[i], x[i])
        x_succ = x[i] + h * 0.5 * (func(points[i], x[i]) + func(points[i+1], x[i+1]))
        while np.sum(abs(x_succ - x[i+1])) > e:
            x[i+1] = x_succ
            x_succ = x[i] + h * 0.5 * (func(points[i], x[i]) + func(points[i+1], x[i+1]))
        x[i+1] = x_succ
    return x


# COUPLED OSCILLATOR
class coupled_oscillator:
    ''' 
    
    Class representing a coupled oscillator system.
    
    Assumptions:
        - fixed: positions of fixed points
        - masses: masses of moving points
        - springs: spring constants and connections
        
    
    '''
    def __init__(self, fixed, masses, springs):
        '''
        
        Parameters:
        ----------
        fixed :
            TYPE: np.ndarray
            DESCRIPTION: array of fixed point positions
        masses :
            TYPE: np.ndarray
            DESCRIPTION: array of masses of moving points
        springs :
            TYPE: np.ndarray
            DESCRIPTION: array of spring constants and their connections
            
        '''
        self.fixed = fixed
        self.masses = masses
        self.springs = springs
        self.n_fisse = len(fixed)               # number of fixed vertices
        self.n_mobili = len(masses)             # number of moving vertices

    def __call__(self, points, x):   
        ''' 
        
        Calculate the state of the system at given time points.
        
        Input:
            - points: vector of time points
            - x: initial state matrix
        
        Output:
            - eq_acc: matrix of state at each time point
            
        '''
        eq_acc = np.zeros((self.n_mobili, 4))        
        eq_acc[:,0:2] = x[:,2:4]                     
        eq_acc[:,2:4] = self.acceleration(x[:,0:2]) 
        return eq_acc
        

# Adjacency list
class list_adjacent(coupled_oscillator):
    ''' 
    
    Class representing a coupled oscillator system using an adjacency list.
    
    '''
    def __init__(self, fixed, masses, springs):
        '''
        
        Parameters:
        ----------
        fixed :
            TYPE: np.ndarray
            DESCRIPTION: array of fixed point positions
        masses :
            TYPE: np.ndarray
            DESCRIPTION: array of masses of moving points
        springs :
            TYPE: list
            DESCRIPTION: adjacency list of spring constants and their connections
            
        '''
        self.fixed = fixed
        self.masses = masses
        self.springs = springs    
        self.n_fisse = len(fixed) 
        self.n_mobili = len(masses)
    
    def acceleration(self, mobili):
        ''' 
        
        Calculate the acceleration of moving points.
        
        Input:
            - mobili: array of positions of moving points
        
        Output:
            - eq_acc: array of accelerations of moving points
            
        '''
        pos = np.concatenate((mobili, self.fixed), axis = 0)  
        eq_acc = np.zeros((self.n_mobili, 2))
        for i in range(self.n_mobili):          
            for (j, kmolla) in self.molle[i]:   
                eq_acc[i] += kmolla * (pos[j] - pos[i]) 
            eq_acc[i] /= self.masses[i] 
        return eq_acc



# Matrix adjacency
class matrix_adjacent(coupled_oscillator):
    ''' 
    
   Class representing a coupled oscillator system using an adjacency matrix.
   
   '''
    def __init__(self, fixed, masses, springs):
        '''
        
        Parameters:
        ----------
        fixed :
            TYPE: np.ndarray
            DESCRIPTION: array of fixed point positions
        masses :
            TYPE: np.ndarray
            DESCRIPTION: array of masses of moving points
        springs :
            TYPE: np.ndarray
            DESCRIPTION: adjacency matrix of spring constants and their connections
            
        '''
        self.fixed = fixed
        self.masses = masses
        self.springs = springs    
        self.n_fisse = len(fixed) 
        self.n_mobili = len(masses) 
        
    def acceleration(self, mobili):
        ''' 
        
        Calculate the acceleration of moving points.
        
        Input:
            - mobili: array of positions of moving points
        
        Output:
            - eq_acc: array of accelerations of moving points
            
        '''
        pos = np.concatenate((mobili, self.fixed), axis=0)
        forces = (self.springs@pos - np.reshape(np.sum(self.springs, axis=1), (self.n_mobili, 1)) * mobili)
        masses = np.reshape(self.masses, (self.n_mobili, 1))
        eq_acc = forces / masses
        return eq_acc                



# Incidence list
class list_incidence(coupled_oscillator):
    ''' 
    
    Class representing a coupled oscillator system using an incidence list.
    
    '''
    def __init__(self, fixed, masses, springs):
        '''
        
        Parameters:
        ----------
        fixed :
            TYPE: np.ndarray
            DESCRIPTION: array of fixed point positions
        masses :
            TYPE: np.ndarray
            DESCRIPTION: array of masses of moving points
        springs :
            TYPE: list
            DESCRIPTION: incidence list of spring constants and their connections
            
        '''
        self.fixed = fixed
        self.masses = masses
        self.springs = springs    
        self.n_fisse = len(fixed) 
        self.n_mobili = len(masses) 
    
    def acceleration(self, mobili):
        ''' 
        
        Calculate the acceleration of moving points.
        
        Input:
            - mobili: array of positions of moving points
        
        Output:
            - eq_acc: array of accelerations of moving points
            
        '''
        pos = np.concatenate((mobili, self.fixed), axis=0)
        eq_acc = np.zeros((self.n_mobili, 2))
        for (i, j, kmolla) in self.springs:
            if i < self.n_mobili:
                eq_acc[i] += kmolla * (pos[j]-pos[i])
            if j <  self.n_mobili:
                eq_acc[j] += kmolla * (pos[i] - pos[j])
        masses = np.reshape(self.masses, (self.n_mobili, 1))
        return eq_acc / masses






# TEST
if __name__ == '__main__':
    
    # Fixed vertex position, I assume chce are not connected by springs
    fixed = np.array([[0.0,0.0],[0.0,5.0],[5.0,5.0],[5.0,0.0]])   
    
    # Initial position moving vertices
    initials_mobile = np.array([[1.0,1.0],[4.0,1.0],[4.0,4.0],[1.0,4.0]])
    
    # Masses
    masses = np.array([1,2,3,4]) 
    
    # Springs: k and how they are connected
    spring_list_adjacent = [[(1,1),(3,1),(4,0.5)],[(0,1),(2,1),(7,0.5)],
                             [(1,1),(3,1),(6,0.5)],[(0,1),(2,1),(5,0.5)]]
    
    springs_matrix_adjacent = np.array([[0,1,0,1,0.5,0,0,0],
                                        [1,0,1,0,0,0,0,0.5],
                                        [0,1,0,1,0,0,0.5,0],
                                        [1,0,1,0,0,0.5,0,0]]) 
    
    springs_list_incidence = [(0,1,1), (0,3,1), (0,4,0.5), (1,2,1), 
                             (1,7,0.5), (2,3,1), (2,6,0.5), (3,5,0.5)]

    
    
    # Functions
    f_list_adjacency = list_adjacent(fixed, masses, spring_list_adjacent)
    f_matrix_adjacent = matrix_adjacent(fixed, masses, springs_matrix_adjacent)
    f_list_incidence = list_incidence(fixed, masses, springs_list_incidence)
    
    # Time
    points_max = 4
    n = 100
    points = np.linspace(start = 0, stop = points_max, num = n)
    
    # Initial state
    initial_state = np.concatenate((initials_mobile, np.zeros((len(masses), 2))), axis=1)
    
    # Numerical schemes applied to functions
    CN = Crank_Nicholson(f_matrix_adjacent, points, initial_state)
    EB = Euler_Backward(f_matrix_adjacent, points, initial_state)
    EF = Euler_Forward(f_matrix_adjacent, points, initial_state)

    
    # PLOTTING
    fig = plt.figure(dpi = 200)
    colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']   # with this list I can represent 8 mobile points
    clr = []
    for i in range(len(masses)):
        clr.append(colour[i])
    for j in range((len(masses))):
        graph = plt.scatter(EF[:,j,0], EF[:,j,1], color = clr[j])
    plt.show()
    
    
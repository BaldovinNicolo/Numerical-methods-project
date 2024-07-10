#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:42:51 2022

@author: nicolobaldovin
"""

import numpy as np
import matplotlib.pyplot as ply


class Library:
    ''' 
    Assunzioni:
        input = matrix
        n-th element of the matrix = function
        explicit method = method in which the next state depends on the derivative of the current instant
        implicit method = method in which the next state depends on the derivative of the next instant
        
    '''
    
    def __init__(self, points, initial_state, e = 1.0e-6):
       '''
      
       Parameters
       ----------
       points :
           TYPE:
               - np.ndarray
           DESCRIPTION:
               - vector whose values ​​are the integration times, which may be non-uniform
               
               
       initial_state: 
           TYPE:
               - np.ndarray
           DESCRIPTION:
               - matrix whose values ​​are the initial conditions for each dimension
               - a matrix must always be provided as input, even in the case of a dimension 
               - the last element I assume is the function
               - the order of the ODE is n and the initial conditions are n
               - structure: [x, x', x'', ..., func]
               - different arrays can have different ODEs and of different orders
               
       e : 
           TYPE:
               - optional
           DESCRIPTION:
               - maximum error in implicit methods
               
               

       Returns
       -------
       None.

       '''
        
       self.points = points                 
       self.initial_state = initial_state
       self.e = e
        
       
        
    def integrator(self, flag, numeric_scheme, max_derivate = 1):
        '''
        
        Parameters
        ----------
        flag : 
            TYPE: 
                - str
            DESCRIPTION:
                - it is used to indicate whether the scheme is explicit or implicit
                - takes as input: "implicit" / "explicit"
                
        numeric_scheme: 
            TYPE:
                - function
            DESCRIPTION:
                - the parameter indicates the integration method
                - schemas can be explicit or implicit
                - the explicit scheme includes the following parameters:
                    i) x0 = value of x in the current state
                    ii) h = interval width
                    iii) derivata_n = vector of derivatives in the current state (x', x'', ...)
                    iv) max_ derivata = maximum number of times you want to derive
                - the implicit scheme includes the following parameters:
                    i) x0 = x value in the current state
                    ii) h = interval width
                    iii) derivata_succ = vector of derivatives in the next state (x'_t+1, x''_t+1, ...)
                    iv) derivata_n = vector of derivatives in the current state (x', x'', ...)                                

        max_derivata : 
            TYPE:
                - optional
            DESCRIPTION:
                - indicates the order
                - a numerical scheme with max_derivative equal to n will be able to calculate the next value with its first n derivatives
                - implemented only in the case flag == "explicit"
        Returns
        -------
        result :
            TYPE:
                - list
            DESCRIPTION:
                - lista of array
                - each array is the set of ordinates assumed by the solution function in that dimension

        '''
        
        result = []
        for i in self.initial_state:   
            self.func = i[-1]
            self.order = len(i) - 1
            
            
            x = np.zeros((len(self.points), len(i)))   # the columns are x, x', x'', ... ; the lines are the moments
            x[0][0] = i[0]
            i[-1] = self.func(x[0][0])
            x[0] = i                                   # x = matrix whose first row is the initial state
            
            
            for j in range(x.shape[0] - 1):      # cycle over the various moments; I start from the first state, because I already know everything about state 0
                h = self.points[j+1] - self.points[j]   
                x_stima = np.zeros(1)            
                x_stima[0] = self.deriv(x[j], 2) 
                x[j+1] = self.update(x[j], h, self.Taylor(x[j][1], h, x_stima), flag, numeric_scheme, max_derivate)
            result.append(x[:,:1])              
            i[-1] = self.func                    # restore the function to the last position
        return result




    def deriv(self, x, n):
        ''' 
        
        It takes as input the current state and extrapolates the derivative of a certain degree n
        It can go beyond the ODE order
        
        '''
        if n == self.order:
            return self.func(x[n - self.order])    # if I'm at the last level I solve with the function
        if n < self.order:
            return x[n]                             
        if n > self.order:
            if n - self.order < self.order:
                return self.func(x[n - self.order]) 
            else:
                return self.func(self.deriv(x, n - self.order)) 


    
    def update(self, x, h, stima, flag, numeric_scheme, max_derivate = 1):
        ''' 
        
        Input:
            - x = current state
            - h = interval width
            - stima = state following the current one calculated with ef for implicit methods
            - flag = indicates the type of scheme ("explicit" / "implicit")
            - schema_numerico = indicates the numerical scheme chosen
            - max_derivata = maximum number of times you want to derive
        Output:
            - x_succ = next state
            
        '''
        
        if flag == "explicit":
            x_succ = np.zeros(x.shape)      
            for i in range(0, self.order + 1):
                derivate = np.zeros(max_derivate + 1)
                for j in range(0, max_derivate + 1):
                    derivate[j] = self.deriv(x, j+i+1)
                if i == self.order:
                    x_succ[i] = self.func(x_succ[0])    
                else:
                    x_succ[i] = numeric_scheme(x[i], h, derivate, max_derivate)
            x_succ[-1] = self.func(x_succ[0])
            return x_succ
        
        if flag == "implicit":
            x_succ = np.zeros(x.shape)
            epsilon = 1    
            stima_vett = np.ones(1)
            stima_vett[0] = stima
            x_succ[0] = self.Taylor(x[0], h, stima_vett) 
            x_succ[-1] = self.func(x_succ[0])   
            
            while (abs(epsilon) > self.e):      # Metodo del punto fisso
                x_n = x_succ[0]
                for i in range(self.order, 0, -1):
                    x_succ[i-1] = numeric_scheme(x[i-1], h, x_succ[i], x[i])
                epsilon = x_succ[0] - x_n
                x_succ[-1] = self.func(x_succ[0])
            return x_succ

     
        
     
    def Taylor(self, x0, h, derivata, max_derivate = 1):
        ''' 
        
        Explicit Taylor series at term n
        
        '''
        x1 = x0
        for i in range(1, max_derivate + 1):
            x1 += derivata[i-1] * (h**(i)) / (np.math.factorial(i))     
        return x1
    
    
    # Implementation for the studied schemes
    def ef(self, max_derivate = 1):    # Eulero Forward
        return self.integrator("esplicito", self.Taylor, max_derivate)
        
    def eb(self):                      # Eulero Backward
        def eb_metodo(x0, h, derivata_succ, derivata_n):
            return x0 + h * derivata_succ
        return self.integrator("implicito", eb_metodo)
        
    def cn(self):                       # Crank Nicholson
        def cn_metodo(x0, h, derivata_succ, derivata_n):
            return x0 + h * (derivata_n + derivata_succ) * 0.5
        return self.integrator("implicito", cn_metodo)

    
    # Method for calling the studied patterns
    def __call__(self):
        '''
        This method allows you to call all 3 default integration methods

        A tensor with 3 matrices is returned: one matrix for ef[0], one for eb[1] and one for cn[2]
        This matrix has in each row the solution vector for that dimension
            
        '''
        return self.ef(), self.eb(), self.cn()
    

if __name__ == '__main__':
    
    def exponential(u):
        return -u
    
    def function(u):
        return -25*u
    
    
    def metodoexp(x0, h, derivata_n, max_derivata = 1): 
        return x0 + h * derivata_n[0]
    
    def metodoimp(x0, h, derivata_succ, derivata_n):    
        return x0 + h * (derivata_succ + derivata_n) / 4
    
    # Integration times
    points_max = 5
    n = 3000
    points = np.linspace(start = 0, stop = points_max, num = n)
    
    # Initial codes
    initial_conditions = np.array([[7, 8,function],[7.5,exponential], [3,4,function]], dtype = object)
    
    # Function
    func = Library(points, initial_conditions)
    
    # Position
    result = func.integrator("implicito", metodoimp)
    
    
    # Implementation for the three default schemes
    resultef = func.ef()
    resulteb = func.eb()
    resultcn = func.cn()
    
    
    fig = ply.figure(dpi = 200)
    colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    clr = []
    for i in range(len(initial_conditions)):
        clr.append(colour[i])
    for j in range(len(initial_conditions)):
        graph = ply.scatter(points, result[j], color = clr[j], s=1)

    ply.show()
 
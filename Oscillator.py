#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:07:47 2022

@author: nicolobaldovin
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
t_start = 0         # initial time
t_end = 10          # final time
n = 366             # number subintervals
h = t_end / (n-1)   # width of each subinterval
k_m = 4             # constant of spring (k/m) 

   
x = np.linspace(start = t_start, stop = t_end, num = n) 
function = np.cos(-(np.sqrt(k_m))*x)              # sol. of ODE


# Eulero Forward
ef = np.zeros(n)
ef[0] = 1           # initialization
derivate_ef = np.zeros(n)
derivate_ef[0] = 0  # initialization

for j in range(0,n-1,1):
    ef[j+1] = ef[j] + h * derivate_ef[j]
    derivate_ef[j+1] = derivate_ef[j] - k_m *h*ef[j]

    
    
# Eulero Backword
eb = np.zeros(n)
eb[0] = 1           # initialization
derivate_eb = np.zeros(n)
derivate_eb[0] = 0  # initialization

for j in range(0,n-1,1):
    eb[j+1] = (eb[j] + h*derivate_eb[j]) / (1 + k_m*h*h)
    derivate_eb[j+1] = (derivate_eb[j] - k_m*h*eb[j]) / (1 + k_m*h*h)
    

# Crank Nicholson
cn = np.zeros(n)
cn[0] = 1           # initialization
derivate_cn = np.zeros(n)
derivate_cn[0] = 0  # initialization

for j in range(0,n-1,1):
    cn[j+1] = ((4-h*h*k_m)*cn[j] + 4*h*derivate_cn[j])/(4+h*h*k_m)
    derivate_cn[j+1] = ((4-h*h*k_m)*derivate_cn[j] - 4*h*k_m*cn[j])/(4+h*h*k_m)



# Plot
fig = plt.figure(dpi = 200)
exact_function = plt.plot(x, function, color = "yellow", label = "Exact Function")
eulero_forward = plt.scatter(x, ef, s = 1, label = "Eulero Forward")
eulero_backward = plt.scatter(x, eb, s = 1, label = "Eulero Backward")
crank_nicholson = plt.scatter(x, cn, s = 1, label = "Crank Nicholson")
plt.legend(prop={'size': 6})
plt.show()



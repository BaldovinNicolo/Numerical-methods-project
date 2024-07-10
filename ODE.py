#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:47:46 2022

@author: nicolobaldovin
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(u):
  return 1 / (1 + np.exp(-u))



def Eulero_Forward(func, points, X):
    n = len(points)
    u = np.zeros(n)     
    u[0] = X           
    for j in range(n-1):
        h = points[j+1] - points[j]
        u[j+1] = u[j] + (h * func(points[j], u[j]))
    return u


def secanti(x0, x1, funz, e = 1.0e-6, max_iter = 100):
    iter = 0
    while abs(x1 -x0) > e and iter < max_iter:
        iter += 1
        x1 = x1 - (funz(x1) * (x1 - x0) / (funz(x1) - funz(x0)))
        x0 = x1
    return x1


def Eulero_Backward(func, points, X):
    n = len(points)
    u = np.zeros(n)
    u[0] = X
    h1 = points[1] - points[0]
    u[1] = u[0] + (h1 * func(points[0], u[0]))
    for j in range(n-2):
        h = points[j+1] - points[j]
        gf = lambda x: u[j+1] + h * func(points[j+2], x) - x
        u[j+2] = secanti(u[j], u[j+1], gf)
    return u



def Crank_Nicholson(func, points, X):
    n = len(points)
    u = np.zeros(n)
    u[0] = X
    h1 = points[1] - points[0]
    u[1] = u[0] + (h1 * func(points[0], u[0]))
    for j in range(n-2):
        h = points[j+1] - points[j]
        gf = lambda x: u[j+1] - x + h * 0.5 * (func(points[j+2], x) + func(points[j+1], u[j+1]))
        u[j+2] = secanti(u[j], u[j+1], gf)
    return u



if __name__ == '__main__':
    end = 3
    n = 80
    X = 1
    func = lambda t,u:  -u 
    points = np.linspace(start = 0,  stop = end,  num = n)
    EF = Eulero_Forward(func, points, X)
    EB = Eulero_Backward(func, points, X)
    CN = Crank_Nicholson(func, points, X)
    
    # Plot
    fig = plt.figure(dpi = 200)
    exact_function = plt.plot(points, np.exp(-points), color = "yellow")
    forwrd = plt.scatter(points, EF, s=1, label = 'Eulero Forward')
    backwrd = plt.scatter(points, EB, s=1, label = 'Eulero Backward')
    cranknich= plt.scatter(points, CN, s=1, label = 'Crank Nicholson')
    plt.legend()
    plt.show()
    

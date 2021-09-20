# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:59:58 2020

@author: ZongSing_NB

Main reference:
https://doi.org/10.1016/j.asoc.2019.105937
"""

import numpy as np
import matplotlib.pyplot as plt

class EWOA():
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0,
                 b=1, a_max=2, a_min=0, a2_max=-1, a2_min=-2, l_max=1, l_min=-1):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub
        self.lb = lb
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b
        
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)
        
    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        
        # 迭代
        for g in range(self.G):
            # 適應值計算
            F = self.fitness(self.X)
            
            # 更新最佳解
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
            
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F
            
            # 更新
            a2 = self.a2_max - (self.a2_max-self.a2_min)*(g/self.G)
            
            for i in range(self.P):
                p = np.random.uniform()
                R2 = np.random.uniform()
                C = 2*R2
                l = (a2-1)*np.random.uniform() + 1
                self.b = np.random.randint(low=0, high=500)
                
                if p<0.5:
                    Dnew = C*self.gbest_X - self.X[i, :] # (4)
                    self.X[i, :] = self.gbest_X  - Dnew*np.cos(2*np.pi*l) # (4)
                else:
                    D = np.abs(self.gbest_X - self.X[i, :]) # (5)
                    self.X[i, :] = self.gbest_X + D*np.exp(self.b*l)*np.cos(2*np.pi*l) # (5)
            
            # 邊界處理
            self.X = np.clip(self.X, self.lb, self.ub)

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.loss_curve[-1], 3))+']')
        plt.plot(self.loss_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
            
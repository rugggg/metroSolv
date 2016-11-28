import numpy as np
import random

numStations = 10
m = np.full((numStations,numStations),-10)

tube = np.genfromtxt('londonUnderground.csv',delimiter=',')
print(tube)



col = tube[:,0]

for i in range (0,len(col)):
    if col[i] == -1:
        col[i] = 10

print(tube)


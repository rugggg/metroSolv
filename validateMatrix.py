import numpy as np
import random

numStations = 10
m = np.full((numStations,numStations),-10)


def validateMatrix(m):
    valid = True
    for i in range (0,len(m[0])):
        col = tube[:,i]
        row  = tube[i,:]
        for j in range(0,len(m[0])):
            if col[j] != row[j]:
                    print("Invalid!")
                    print("Row/Column ",i,j)
                    print("    row: ",m[i,j])
                    print("    col: ",m[j,i])

tube = np.genfromtxt('londonUnderground.csv',delimiter=',')
#tube = np.genfromtxt('small.csv',delimiter=',')


print(tube)

validateMatrix(tube)



col = tube[:,0]
row  = tube[0,:]


for i in range (0,len(col)):
    if col[i] == -1:
        col[i] = 10
    if row[i] == -1:
        row[i] = 10

print(tube)


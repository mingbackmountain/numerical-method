
# coding: utf-8

# In[99]:


#Thanakorn Pasangthien 6088109 Section1
##### Question A
from math import e
import numpy as np
from numpy import sign

def f(m,x):
    return m-np.log(x)
def falseposition(m,xl,xu):
    xr = 0
    xr_prev = 0
    ea = 100
    es = 0.005
    while (ea > es):
        xr_prev = xr
        f_xu = f(m,xu)
        f_xl = f(m,xl)
        xr = xu - ( f_xu * (xu - xl) / ( f_xu - f_xl ) )
        f_xr = f(m,xr)
        if f_xl*f_xr <= 0:
            xu = xr
        else:
            xl = xr
        ea = abs(100*(xr-xr_prev)/xr)
    return xr

print("e^x = "+str(falseposition(2,1,5)))


# In[25]:


#Question B
import math

def f(x):
    return (x**3)-(6*(x**2))+x-4

def g(x):
    return 3*(math.sin(x))+(math.cos(x))**2

# Golden Section Search
def goldSearch(func,xl,xu):
    es = 0.005
    iterate = 0
    max_iterate = 20
    phi = 1.6180
    xopt = 0
    if func == 0:
        while True:
            d = (phi-1)*(xu-xl)
            x1 = xl + d
            x2 = xu - d
            if f(x1) < f(x2):
                xopt = x1
                xl = x2
            else:
                xopt = x2
                xu = x1
            iterate += 1
            ea = abs((2-phi)*(xl-xu)/xopt)*100
            if ((ea <= es) or iterate >= max_iterate):
                break;
    else:
         while True:
            d = (phi-1)*(xu-xl)
            x1 = xl + d
            x2 = xu - d
            if g(x1) < g(x2):
                xopt = x1
                xl = x2
            else:
                xopt = x2
                xu = x1
            iterate += 1
            ea = abs((2-phi)*(xl-xu)/xopt)*100
            if ((ea <= es) or iterate >= max_iterate):
                break;
    return xopt

result_fx = f(goldSearch(0,0,6))
result_gx = g(goldSearch(1,math.pi,(2*math.pi)))
print("Mininum Golden Section Search")
print("optimize: x = {:.4f} f(x) = {:.4f}".format(goldSearch(0,0,6),result_fx))
print("optimize: x = {:.4f} g(x) = {:.4f}".format(goldSearch(1,math.pi,(2*math.pi)),result_gx))
print("-----------------------------------------------")

#parabolic interpolation
def parabolic(func,x1,x2,x3):
    max_iterate = 20
    es = 0.005
    iterate = 0
    xopt = 0
    xopt_prev = 0
    if func == 0:
        while True:
            xopt_prev = xopt
            alpha1 = (x2-x1)*(x2-x1)*(f(x2)-f(x3))
            alpha2 = (x2-x3)*(x2-x3)*(f(x2)-f(x1))
            beta1  = (x2-x1)*(f(x2)-f(x3))
            beta2  = (x2-x3)*(f(x2)-f(x1))
           
            gamma = (alpha1 - alpha2)/(beta1 - beta2)
            x4 = x2 - (0.5 * gamma)

            if x4 > x2:
                x1 = x2
                x2 = x4
            else:
                x3 = x2
                x2 = x4
            xopt = x4
            iterate += 1
            ea = abs((xopt - xopt_prev)/xopt)*100
            if(ea <= es or iterate >= max_iterate):
                break
    else:
         while True:
            xopt_prev = xopt
            alpha1 = (x2-x1)*(x2-x1)*(g(x2)-g(x3))
            alpha2 = (x2-x3)*(x2-x3)*(g(x2)-g(x1))
            beta1  = (x2-x1)*(g(x2)-g(x3))
            beta2  = (x2-x3)*(g(x2)-g(x1))
           
            gamma = (alpha1 - alpha2)/(beta1 - beta2)
            x4 = x2 - (0.5 * gamma)

            if x4 > x2:
                x1 = x2
                x2 = x4
            else:
                x3 = x2
                x2 = x4
            xopt = x4
            iterate += 1
            ea = abs((xopt - xopt_prev)/xopt)*100
            if(ea <= es or iterate >= max_iterate):
                break
    return xopt
print("Maximum Parabolic Interpolation")
result_Pfx = f(parabolic(0,-2,0,2))
result_Pgx = g(parabolic(1,(-math.pi/2),0,math.pi))
print("optimize: x = {:.4f} f(x) = {:.4f}".format(parabolic(0,-2,0,2),result_Pfx))
print("optimize: x = {:.4f} g(x) = {:.4f}".format(parabolic(1,(-math.pi/2),0,math.pi),result_Pgx))
print("-----------------------------------------------")


# In[26]:


#Question C
import math
import numpy as np
import scipy 
import scipy.linalg
import numpy.linalg as m

#part 1 determinant without recursive
def det(matrix):
    Det = 1
    row = len(matrix)
    colum = len(matrix[0])
    P,L,U = scipy.linalg.lu(matrix)
    det_p = m.det(P)
    det_l = m.det(L)
    for i in range(row):
        for j in range(colum):
            if i == j:
                Det *= U[i,j]
    Det = ((Det*det_l)/(det_p))
    return Det
    
M = [[1,3,-2],
     [3,2,6],
     [2,4,3]
    ]

C = [[1,3,-2,4],
     [3,2,6,2],
     [2,4,3,1],
     [2,4,3,5]]

print("Determinant 3X3 = "+str(det(M)))
print("Determinant 4X4 = "+str(det(C)))
print("-----------------------------------")

#Part 2 Carmer'Rule
m1 = np.asmatrix(M)
m2 = np.asmatrix(M)
m3 = np.asmatrix(M)
detM = det(M)

m1[:,0] = [[1],[-2],[2]]
m2[:,1] = [[1],[-2],[2]]
m3[:,2] = [[1],[-2],[2]]

m1 = np.asarray(m1)
m2 = np.asarray(m2)
m3 = np.asarray(m3)

detm1 = det(m1)
detm2 = det(m2)
detm3 = det(m3)
print("Carmer'Rule to solve equation")
print("a = "+str(detm1/detM))
print("b = "+str(detm2/detM))
print("c = "+str(detm3/detM))
print("-----------------------------------")

#Part 3 Find Inverse
def transposeMatrix(m):
    result = np.zeros((3,3),dtype=float)
    for i in range(len(m)):
        for j in range(len(m[0])):
            result[j][i] = m[i][j]
    return result

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]


def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def inverse(m):
    det = getMatrixDeternminant(m)
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m[0])):
            minor = getMatrixMinor(M,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors[0])):
            cofactors[r][c] = cofactors[r][c]/det
    return cofactors

print("Inverse Matrix")
print(inverse(M))


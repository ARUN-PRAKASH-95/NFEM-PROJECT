import numpy as np
import matplotlib.pyplot as plt
step = 0.075
t=np.arange(step,1,step)

poissons_ratio = 0.3
E = 0.1
mu = E/(2*(1+poissons_ratio))
Lambda=poissons_ratio*E/((1-2*poissons_ratio)*(1+poissons_ratio))
volumetric_strain = -0.01
yield_stress = 100e-6 #N/micromm^2
Gauss_weight = 2
# Mesh generation
Inner_radius = 25
Outer_radius = 100
n=36
meshrefinementfactor = 5
q=meshrefinementfactor**(1/(n-1))

l=(Outer_radius-Inner_radius)*(1-q)/(1-meshrefinementfactor*q)
rnode=Inner_radius
coordinate=np.array([Inner_radius])

for i in range(n):
        rnode=rnode+l
        coordinate=np.append(coordinate,rnode)
        l=l*q

C = np.array([[Lambda+2*mu,Lambda,Lambda],
              [Lambda,Lambda+2*mu,Lambda],
              [Lambda,Lambda,Lambda+2*mu]])

#Tangent stiffness parameters
a=np.eye(3)
c=np.outer(a,a)
Kay = 3*Lambda+2*mu
b=np.eye(9)
d=np.zeros((9,9))
d[0,0]=1; d[1,3]=1;d[2,6]=1;d[3,1]=1;d[4,4]=1;d[5,7]=1;d[6,2]=1;d[7,5]=1;d[8,8]=1
I = (1/2*(b+d))-(1/3*c)
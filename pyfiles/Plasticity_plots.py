import numpy as np
import matplotlib.pyplot as plt

step = 0.01
t=np.arange(step,1,step)

poissons_ratio = 0.2
E = 0.2 
mu = E/(2*(1+poissons_ratio))
Lambda=poissons_ratio*E/((1-2*poissons_ratio)*(1+poissons_ratio))
volumetric_strain = -0.01
yield_stress = 200e-6 #N/micromm^2

# Mesh generation
Inner_radius = 5
Outer_radius = 20
n=10
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


Global_F_ext = np.zeros((n+1,1))
Global_plastic_strain = np.zeros((n,3,1))
Global_displacement = np.zeros((n+1,1)) 
Reduced_displacement = np.delete(Global_displacement,(0),axis=0)
sigma_rr_ri = []
fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(10,5))
for time,tau in enumerate(t):
    Global_displacement[0]=1/3*tau*(-volumetric_strain)*Inner_radius
    print((time+1),''+'-'*100)
    delta_u = np.array([1,1])
    G_red = np.array([1,1])
    Global_F_int = np.array([1,1])
    plastic_strain = np.zeros((n,3,1))
    sigma_rr = np.zeros_like(coordinate)
    sigma_phi = np.zeros_like(coordinate)
   
    while np.linalg.norm(delta_u,np.inf)>(0.005*np.linalg.norm(Reduced_displacement,np.inf)) or np.linalg.norm(G_red,np.inf)>(0.005*np.linalg.norm(Global_F_int,np.inf)) :
        Global_K = np.zeros((n+1,n+1))
        Global_F_int= np.zeros((n+1,1))
        print("NEWT")
        for i in range(len(coordinate)-1):

            def element_routine(coordinate,Lambda,mu,tau):
                Derivative_N = np.array([-1/2,1/2])
                Jacobian = Derivative_N@np.array([[coordinate[i]],
                                                  [coordinate[i+1]]])
                J_inv = np.asscalar(1/Jacobian)

                B = np.array([[-1/2*J_inv,1/2*J_inv],
                              [1/(coordinate[i]+coordinate[i+1]),1/(coordinate[i]+coordinate[i+1])],
                              [1/(coordinate[i]+coordinate[i+1]),1/(coordinate[i]+coordinate[i+1])]])
                B_T = np.transpose(B)
                C_matrix,I_stress = material_routine(coordinate)
                Element_stiffness = 2*(B_T@C_matrix@B)*Jacobian*((coordinate[i]+coordinate[i+1])/2)**2
                Internal_force = 2*(B_T@I_stress)*Jacobian*((coordinate[i]+coordinate[i+1])/2)**2
                return Element_stiffness,Internal_force
            


            def material_routine(coordinate):
                Derivative_N = np.array([-1/2,1/2])
                Jacobian = Derivative_N@np.array([[coordinate[i]],
                                                  [coordinate[i+1]]])
                J_inv = np.asscalar(1/Jacobian)

                B_matrix = np.array([[-1/2*J_inv,1/2*J_inv],
                              [1/(coordinate[i]+coordinate[i+1]),1/(coordinate[i]+coordinate[i+1])],
                              [1/(coordinate[i]+coordinate[i+1]),1/(coordinate[i]+coordinate[i+1])]])
           
                Current_strain = B_matrix@np.array([Global_displacement[i],Global_displacement[i+1]]) 
                Trial_stress = C@(Current_strain - Global_plastic_strain[i])
                Trial_dev = Trial_stress - 1/3*np.sum(Trial_stress)
                Trial_eq = np.sqrt(3/2*(np.sum(np.square(Trial_dev))))
                dev_tensor = np.diagflat(Trial_dev)
                dev_outer = np.outer(dev_tensor,dev_tensor)

                if Trial_eq - yield_stress < 0:
                    sigma_rr[i]=np.asscalar(Trial_stress[0])
                    sigma_phi[i]= np.asscalar(Trial_stress[1])
                    print('ELASTIC',tau,i)
                    return C,Trial_stress
                else:
                    del_lamda = (Trial_eq - yield_stress)/(3*mu)
                    print('PLASTIC',tau,i)
                    plastic_strain[i] = Global_plastic_strain[i] + del_lamda * 1.5 * (Trial_dev/Trial_eq)
                    new_stress = C@(Current_strain-plastic_strain[i])
                    sigma_rr[i]=np.asscalar(new_stress[0])
                    sigma_phi[i]= np.asscalar(new_stress[1])
                    new_dev = new_stress - 1/3*np.sum(new_stress)
                    new_eq = np.sqrt(3/2*(np.sum(np.square(new_dev))))
                    tangent_stiffness = Kay/3*(c)+2*mu*((Trial_eq-3*mu*del_lamda)/Trial_eq)*I-(3*mu*(1/Trial_eq**2)*dev_outer)
                    skip = tangent_stiffness[0::4]
                    C_t = skip[np.nonzero(skip)].reshape(3,3)
                    return C_t,new_stress

            Ke,Fe_int = element_routine(coordinate,Lambda,mu,tau)

            #Assignmnet matrix
            Ae = np.zeros((2,n+1))
            Ae[0,i]=1                                            # [0,n-1]=1 [1,n]=1
            Ae[1,i+1]  =1
            AeT = np.transpose(Ae)


            #Global stiffness matrix
            K=AeT@Ke@Ae                                          # Stiffness matrix  of each element after transformation
            Global_K = np.add(Global_K,K)  
            #Internal force
            F_int = AeT@Fe_int
            Global_F_int = np.add(Global_F_int,F_int)

        K_red = np.delete(Global_K,(0),axis=0)
        K_red = np.delete(K_red,(0),axis=1)

        #Newton Raphson method

        G_matrix = Global_F_int - Global_F_ext
        G_red = np.delete(G_matrix,(0),axis=0)
        delta_u = np.linalg.inv(K_red)@G_red
        Reduced_displacement = Reduced_displacement - delta_u
        Global_displacement = np.insert(Reduced_displacement,(0),(1/3*(tau)*(-volumetric_strain)*Inner_radius)).reshape(n+1,1)
        
    sigma_rr_ri.append(sigma_rr[0])
    Global_plastic_strain=plastic_strain
    
ax[0,0].plot(coordinate,sigma_rr)
ax[0,0].set_title('Sigma_rr Stress Plot',pad=20)
ax[0,0].set_xlabel('Radius')
ax[0,0].set_ylabel('Sigma_rr')
ax[0,1].plot(coordinate,Global_displacement)
ax[0,1].set_title('Displacement Plot')
ax[0,1].set_xlabel('Radius')
ax[0,1].set_ylabel('Displacement')
ax[1,0].plot(coordinate,sigma_phi)
ax[1,0].set_title('Sigma_phi Stress plot')
ax[1,0].set_xlabel('Radius')
ax[1,0].set_ylabel('Sigma_phi')
ax[1,1].plot(t,sigma_rr_ri)
ax[1,1].set_title('Stress at inner most node')
ax[1,1].set_xlabel('No of iterations')
ax[1,1].set_ylabel('sigma_rr_ri')
plt.tight_layout()
# plt.show()
plt.savefig('plots.png')


   
   
    
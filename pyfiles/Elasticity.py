import numpy as np
import matplotlib.pyplot as plt
import itertools

tau= 0.01
poissons_ratio = 0.3
E = 0.1
Initial_stress = 200     #MPa
Inner_radius = 1
Outer_radius = 100
Epsilon = 0              #Gauss point
delta_u = 1

no_of_elements = np.array([30])
Shape_func = np.array([1/2*(1-Epsilon),1/2*(1+Epsilon)])
mu = E/(2*(1+poissons_ratio))
Lambda=poissons_ratio*E/((1-2*poissons_ratio)*(1+poissons_ratio))
volumetric_strain = -0.01


fig,ax = plt.subplots(ncols=2,figsize=(10,5))

R = np.linspace(1,100,100)
u_elastic = ((Inner_radius)**3*(-volumetric_strain)*tau)/(3*(R)**2)
sigma_rr_el = (-2*(E*tau*-volumetric_strain)*Inner_radius**3)/(3*(1+poissons_ratio)*(R**3))
ax[0].plot(R,u_elastic,'k',label='Exact')
ax[1].plot(R,sigma_rr_el,'k',label='Exact')
marker = itertools.cycle((',', '+', '.', 'o', '*','1'))
#displacement_1 = 1/3*tau*volumetric_strain*Inner_radius
C = np.array([[Lambda+2*mu,Lambda,Lambda],
              [Lambda,Lambda+2*mu,Lambda],
              [Lambda,Lambda,Lambda+2*mu]])

for n in no_of_elements:
    meshrefinementfactor = 5
    q=meshrefinementfactor**(1/(n-1))

    l=(Outer_radius-Inner_radius)*(1-q)/(1-meshrefinementfactor*q)
    rnode=Inner_radius
    coordinate=np.array([Inner_radius])
    coordinate=np.array([Inner_radius])

    for i in range(n):
            rnode=rnode+l
            coordinate=np.append(coordinate,rnode)
            l=l*q


    Global_displacement = np.zeros((n+1,1)) 
    Global_displacement[0]= 1/3*tau*(-volumetric_strain)*coordinate[0]

    Reduced_displacement = np.delete(Global_displacement,(0),axis=0)
    Global_F_ext = np.zeros((n+1,1))

    stress=np.zeros_like(coordinate-1)

    while np.linalg.norm(delta_u)>(0.005*np.linalg.norm(Reduced_displacement)):
        Global_K = np.zeros((n+1,n+1))  
        for i in range(len(coordinate)-1):  #Nodes-1 no of elements

            def element_routine(coordinate):
                Derivative_N = np.array([-1/2,1/2])
                Jacobian = Derivative_N@np.array([[coordinate[i]],
                                                  [coordinate[i+1]]])
                J_inv = np.asscalar(1/Jacobian)

                B = np.array([[-1/2*J_inv,1/2*J_inv],
                              [1/(coordinate[i]+coordinate[i+1]),1/(coordinate[i]+coordinate[i+1])],
                              [1/(coordinate[i]+coordinate[i+1]),1/(coordinate[i]+coordinate[i+1])]])

                B_T = np.transpose(B)
                Element_stiffness = 2*(B_T@(material_routine(Lambda,mu))@B)*Jacobian*((coordinate[i]+coordinate[i+1])/2)**2

                #External force calculation
                sigma_rr = 2*mu*J_inv*((-Global_displacement[i]+Global_displacement[i+1])/2)+Lambda*tau*volumetric_strain

                Fe = np.array([[-np.asscalar(sigma_rr)*coordinate[i]**2],
                               [np.asscalar(sigma_rr)*coordinate[i+1]**2]])
                #print(Fe)
                return Element_stiffness



            def material_routine(Lambda,mu):
                    Derivative_N = np.array([-1/2,1/2])
                    Jacobian = Derivative_N@np.array([[coordinate[i]],
                                                      [coordinate[i+1]]])
                    J_inv = np.asscalar(1/Jacobian)

                    B_matrix = np.array([[-1/2*J_inv,1/2*J_inv],
                                  [1/(coordinate[i]+coordinate[i+1]),1/(coordinate[i]+coordinate[i+1])],
                                  [1/(coordinate[i]+coordinate[i+1]),1/(coordinate[i]+coordinate[i+1])]])
                    Current_strain = B_matrix@np.array([Global_displacement[i],Global_displacement[i+1]])
                    Trial_stress = C@Current_strain
                    stress[i]=np.asscalar(Trial_stress[0])
                    return C

            #Assignmnet matrix
            Ae = np.zeros((2,n+1))
            Ae[0,i]=1                                            # [0,n-1]=1 [1,n]=1
            Ae[1,i+1]  =1
            AeT = np.transpose(Ae)

            Ke = element_routine(coordinate)
            #Global stiffness matrix
            K=AeT@Ke@Ae                                          # Stiffness matrix  of each element after transformation
            Global_K = np.add(Global_K,K)                        # Global stiffnes matrix

        K_red = np.delete(Global_K,(0),axis=0)
        K_red = np.delete(K_red,(0),axis=1)






        Global_F_ext = np.zeros((n+1,1))
        #Newton Raphson method


        G_matrix = Global_K@Global_displacement - Global_F_ext

        G_red = np.delete(G_matrix,(0),axis=0)
        delta_u = np.linalg.inv(K_red)@G_red
        Reduced_displacement = Reduced_displacement - delta_u

        Global_displacement = np.insert(Reduced_displacement,(0),(1/3*tau*(-volumetric_strain)*Inner_radius)).reshape(n+1,1)

    
    
    ax[0].plot(coordinate,Global_displacement,label=str(n)+'elements', marker= next(marker))
    ax[0].set_xlabel('Radius',fontsize=15)
    ax[0].set_ylabel('Displacement',fontsize=15)
    ax[0].legend(loc='upper right', shadow=True, fontsize='x-large')
    ax[1].plot(coordinate,stress,label=str(n)+'elements', marker= next(marker))
    ax[1].set_xlabel('Radius',fontsize=15)
    ax[1].set_ylabel('$\sigma_{rr}$',fontsize=20)
    ax[1].legend(loc='lower right', shadow=True, fontsize='x-large')
    
    fig.tight_layout()
    plt.savefig('Elastic.png')
    
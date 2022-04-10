from MNL import *
import matplotlib.pyplot as plt
A = [[2,-3,0,0,0,0],[-1,4,-1,0,-1,0],[0,-1,4,0,0,-1],[0,0,0,2,-3,0],[0,-1,0,-1,4,-1],[0,0,-1,0,-1,4]]
b = [-5/3,2/3,3,-4/3,-1/3,5/3]
X = a_mat(A,b)

#Plotting Jacobi method residue vs steps
Res1 = []
k1 = Jacobi_Inv(A,b)[2]
Res1.append(Jacobi_Inv(A,b)[1])
x = np.linspace(1,k1,1)
plt.plot(x,Res1)
plt.xlabel('Number of steps')
plt.ylabel('Residue')
plt.title('Residue vs steps plot for Jacobi method')
plt.show()

#Plotting Gauss Seidel residue vs steps
Res2 = []
k2 = Gau_Seid(A,b)[2]
Res2.append(Gau_Seid(A,b)[1])
x2 = np.linspace(1,k2,1)
plt.plot(x2,Res2)
plt.xlabel('Number of steps')
plt.ylabel('Residue')
plt.title('Residue vs steps plot for Gauss-Seidel method')
plt.show()

#Plotting Conjugate Gradient reisdue vs steps
Res3 = []
k3 = ConjuGra(A,b,0.00001)[2]
Res3.append(ConjuGra(A,b,0.00001)[1])
x3 = np.linspace(1,k3,1)
plt.plot(x3,Res3)
plt.xlabel('Number of steps')
plt.ylabel('Residue')
plt.title('Residue vs steps plot for Conjugate Gradient method ')
plt.show()


#To get the Inverse
def Inv(mat, tol=0.00001, plot=False, name="JacobiInv"):
    if (name == "JacobiInv"):
        solv = Jacobi_Inv
    if (name == "Gauss-Seidel"):
        solv = gauss_sidel
    if (name == "ConjugateGrad"):
        solv = ConjGrad

    I = np.identity(len(mat))
    Inv = np.zeros((len(mat), len(mat)))
    for i in range(len(mat)):
        Inv[:, i] = solv(mat, I[i], tol=tol)

    return Inv
#Inverse matrices using Jacobi
print('Solution using Jacobi inverse method is: ',Jacobi_Inv(A,b)[0])
#Solution using Gauss-Seidel method -
print('Solution using gauss-seidel is: ',Gau_Seid(A,b)[0])
#Solution using Conjugate Gradient method -
print('Solution using Conjugate Gradient is: ',ConjuGra(A,b,0.00001))




from MNL import *
# Writing the augmented matrix and storing it in A
A = [[1,-1,4,0,2,9],[0,5,-2,7,8,4],[1,0,5,7,3,-2],[6,-1,2,3,0,8],[-4,2,0,5,-5,3],[0,7,-1,5,4,-2]]
b = [19,2,13,-7,-9,2]
X = a_mat(A,b)
#Gauss-Jordan solution -
print('Solution using Gauss-Jordan is: ',GauJo(X))
#LU Decomposition solution -
print('Solution using LU Decomposition is: ',luD(A,b,6))

#Solution using Gauss-Jordan is:
# [1.1580753585739938, -3.9985768070008967, -0.059642889586468684, 2.298941184782938, -1.0269801055787147, 3.929122969931263]
#Solution using LU Decomposition is:
# [1774.3846153846155, 5617.63076923077, 520.6153846153846, -1.786764705882353, -175.53846153846155, 1.2]

from MNL import *
import matplotlib.pyplot as plt

# Writing A, b for Ax = b
A = [[2, -3, 0, 0, 0, 0], [-1, 4, -1, 0, -1, 0], [0, -1, 4, 0, 0, -1], [0, 0,
    0, 2, -3, 0], [0, -1, 0, -1, 4, -1], [0, 0, -1, 0, -1, 4]]
b = [-5/3, 2/3, 3, -4/3, -1/3, 5/3]

# Solution using LU and Jacobi
sol_lu = lu_decomp(A, b)
sol_j = jacobi(A, b, 1e-4)[0]
sol_gs = gauss_seidel(A, b, 1e-4)[0]

#Solutions
print("Solution using LU Decomposition is : {}".format(sol_lu))
print("Solution using Jacobi method is : {}".format(sol_j))
print("Solution using Gauss-Seidel is : {}".format(sol_gs))

# Calculating inverse of A using Jacobi, Gauss-Seidel and Conjugate Gradient

#Rows of identity matrix
iden1 = [1, 0, 0, 0, 0, 0]
iden2 = [0, 1, 0, 0, 0, 0]
iden3 = [0, 0, 1, 0, 0, 0]
iden4 = [0, 0, 0, 1, 0, 0]
iden5 = [0, 0, 0, 0, 1, 0]
iden6 = [0, 0, 0, 0, 0, 1]

# Inverse using Jacobi method
colj1, itej, resij = jacobi(A, iden1, 1e-4)
colj2 = jacobi(A, iden2, 1e-4)[0]
colj3 = jacobi(A, iden3, 1e-4)[0]
colj4 = jacobi(A, iden4, 1e-4)[0]
colj5 = jacobi(A, iden5, 1e-4)[0]
colj6 = jacobi(A, iden6, 1e-4)[0]

# Combinig the columns in a matrix
colinv = [colj1, colj2, colj3, colj4, colj5, colj6]
#Combing the arrays
mat_inv_j = list(zip(*colinv))
# Inverse of A using Jacobi
print("\nInverse of A using Jacobi method:")
mat_print(mat_inv_j)

# Gauss-Seidel
colgs1, itergs, resigs = gauss_seidel(A, iden1, 1e-4)
colgs2 = gauss_seidel(A, iden2, 1e-4)[0]
colgs3 = gauss_seidel(A, iden3, 1e-4)[0]
colgs4 = gauss_seidel(A, iden4, 1e-4)[0]
colgs5 = gauss_seidel(A, iden5, 1e-4)[0]
colgs6 = gauss_seidel(A, iden6, 1e-4)[0]

Colinv = [colgs1, colgs2, colgs3, colgs4, colgs5, colgs6]
#combining the arrays
mat_inv = list(zip(*Colinv))
# Inverse of A using Gauss-Seidel
print("\nInverse of A using Gauss-Seidel method:")
mat_print(mat_inv)

# Comparing convergence rates of Jacobi and Gauss-Seidel
plt.plot(itej, resij, label="Jacobi method")
plt.plot(itergs, resigs, label="Gauss-Seidel method")
plt.xlabel("Iterations")
plt.ylabel("Residue")
plt.title("Comparing convergence")
plt.legend()
plt.show()




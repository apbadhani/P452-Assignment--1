from MNL import *

# Defining A,b to write Ax = b
A = [[1, -1, 4, 0, 2, 9], [0, 5, -2, 7, 8, 4], [1, 0, 5, 7, 3, -2], [6, -1, 2,
    3, 0, 8], [-4, 2, 0, 5, -5, 3], [0, 7, -1, 5, 4, -2]]
b = [19, 2, 13, -7, -9, 2]

# Solution using gauss-jordan
sol_gj = gau_jor(A, b)
print("Solution using Gauss-Jordan is : {}\n".format(sol_gj))

# Solution using gauss-jordan
sol_lu = lu_decomp(A, b)
print("Solution using LU-decomposition is : {}".format(sol_lu))


from MNL import *
import numpy as np
import matplotlib.pyplot as plt

#Defining kronecker delta function
def kro(a, b):
    return 1 if a == b else 0

# Defining the matrix in the problem
def matri6(x, y, m=0.2, dim=20):
    i, j = x // dim, x % dim
    a, b = y // dim, y % dim

    mat = 0.5 * (
        (kro(i + 1, a) * kro(j, b))
        + (kro(i - 1, a) * kro(j, b))
        - (4 * kro(i, a) * kro(j, b))
        + (kro(i, a) * kro(j + 1, b))
        + (kro(i, a) * kro(j - 1, b))
    ) + (m**2) * kro(i, a) * kro(j, b)
    return mat

#Multiply function with vector
def func_multi(f, x):
    n = len(x)
    prod = np.zeros(n)
    for i in range(n):
        prod[i] = sum([f(i, j) * x[j] for j in range(n)])

    return prod

# Function of Conjugate Gradient
def con_grad_of(func, b, tol):
    n = len(b)
    count = 0
    x = np.zeros(n)
    r = b - func_multi(func, x)
    d = np.copy(r)
    residue = [np.linalg.norm(r)]
    iterations = [count]
    for i in range(n):
        Ad = func_multi(func, d)
        rprevdot = np.dot(r, r)
        alpha = rprevdot / np.dot(d, Ad)
        x += alpha * d
        r -= alpha * Ad
        rnextdot = np.dot(r, r)
        count += 1
        iterations.append(count)
        residue.append(np.linalg.norm(r))

        if np.linalg.norm(r) < tol:
            return x, iterations, residue

        else:
            beta = rnextdot / rprevdot
            d = r + beta * d
            rprevdot = rnextdot


def Inv_mat(A, tol, N=400):
    inv = []
    B = np.identity(N)
    for i in range(3):
        x, iter, residue = con_grad_of(A, B[:, i], tol)
        inv.append(x)

    return np.array(inv).T, iter, residue


x, iter, res = Inv_mat(matri6, 1e-6)
# Printing matrix
print("Inverse of matrix is : ")
print(x)

# Generating plot of convergence
plt.plot(iter, res)
plt.title('Residue vs Iteration plot')
plt.xlabel("Iterations")
plt.ylabel("Residue")
plt.show()

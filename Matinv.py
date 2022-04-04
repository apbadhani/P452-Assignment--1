import numpy as np
import sys
#Creating a zero matrix of order m*n
def zeromatrix(m,n):
        p= [[0 for i in range(n)] for j in range(m)]
        return(p)



#function for matrix vector multiplication
def mat_vec_mult(A, B):
    n = len(B)
    if len(A[0]) == n:
        p = [0 for i in range(n)]
        for i in range(n):
            for j in range(n):
                p[i] = p[i] + (A[i][j] * B[j])
        return (p)
    else:
        print('This combination is not suitable for multiplication')

#Partial Pivot
def Parpivot(A, B, k):
    if np.abs(A[k][k]) < 1e-10:
        n = len(B)
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[k][k]) and abs(A[k][k]) == 0:
                A[k], A[i] = A[i], A[k]
                B[k], B[i] = B[i], B[k]
    return A, B

#Gauss-Jordan
def GaussJordan(A, B):
    n = len(B)
    for k in range(n):
        Parpivot(A, B, k)
        # the pivot row
        pivot = A[k][k]
        # To divide entire pivot row by the pivot
        for i in range(k, n):
            A[k][i] = A[k][i]/pivot

        B[k] = B[k] / pivot
        # other rows
        for i in range(n):
            if abs(A[i][k]) < 1e-10 or i == k:
                continue
            else:
                term = A[i][k]
                for j in range(k, n):
                    A[i][j] = A[i][j] - term * A[k][j]
                B[i] = B[i] - term * B[k]
    return B, A

#LU Decomposition
MAX = 100
def luD(mat,b, n):
    lower = [[0 for x in range(n)]
             for y in range(n)]
    upper = [[0 for x in range(n)]
             for y in range(n)]

    # Decomposing matrix into Upper
    # and Lower triangular matrix
    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            # Summation of L(i, j) * U(j, k)
            sum = 0
            for j in range(i):
                sum += (lower[i][j] * upper[j][k])
            # Evaluating U(i, k)
            upper[i][k] = mat[i][k] - sum
        # Lower Triangular
        for k in range(i, n):
            if (i == k):
                lower[i][i] = 1  # Diagonal as 1
            else:
                # Summation of L(k, j) * U(j, i)
                sum = 0
                for j in range(i):
                    sum += (lower[k][j] * upper[j][i])
                # Evaluating L(k, i)
                lower[k][i] = int((mat[k][i] - sum)/upper[i][i])
    L = lower
    U = upper
    # Performing substitution Ly=b
    y = [0 for i in range(n)]
    for i in range(0, n, 1):
        y[i] = b[i] / float(L[i][i])
        for k in range(0, i, 1):
            y[i] -= y[k] * L[i][k]
    # Performing substitution Ux=y
    x = [0 for i in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = y[i] / float(U[i][i])
        for j in range(i + 1, n):
            x[i] -= x[k] * U[i][k]
    print('Solution using LU Decomposition is :', x)


#tolerance level added here
# Jacobi Method - All eigenvalues. For numpy array
def Jacobi(A,e):
    # largest off-diag element
    n = len(A)
    def maxind(A):
        Amax = 0
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(A[i,j])>=Amax:
                    Amax = abs(A[i,j])
                    k = i
                    l = j
        return Amax, k,l

    # to make A[k,l] = 0 by rotation and define rotation matrix
    def rotate(A,p,k,l):
        A_diff = A[l,l]-A[k,k]
        if abs(A[k,l])< abs(A_diff) * 1e-30:
            t = A[k,l]/A_diff
        else:
            phi = A_diff/(2*A[k,l])
            t = 1/(abs(phi)+np.sqrt(phi**2+1))
            if phi<0:
                t = -t
        c = 1/np.sqrt(t**2+1)
        s = t*c
        tau = s/(1+c)

        term = A[k,l]
        A[k,l] = 0
        A[k,k] = A[k,k] - t*term
        A[l,l] = A[l,l] + t*term
        for i in range(k):
            term = A[i,k]
            A[i,k] = term - s*(A[i,l] + tau*term )
            A[i,l] += s*(term- tau*A[i,l])
        for i in range(k+1,l):
            term = A[k,i]
            A[k,i] = term - s*(A[i,l] + tau*A[k,i])
            A[i,l] += s*(term - tau*A[i,l])
        for i in range(l+1, n):
            term = A[k,i]
            A[k,i] = term - s*(A[l,i] + tau*term)
            A[l,i] += s*(term - tau*A[l,i])
        for i in range(n):
            term = p[i,k]
            p[i,k] = term - s*(p[i,l] - tau*p[i,k])
            p[i,l] += s*(term - tau*p[i,l])

    p = np.identity(n)
    for i in range(4*n**2):
        Amax, k,l = maxind(A)
        if Amax < 1e-9:
            return np.diagonal(A)
        rotate(A,p,k,l)
    print("This method did not converge")

#Gauss Seidel

#tolerance level is added
def gauss_seid_inv(A, B, eps=1.0e-4):
    # Check: A should have zero on diagonals
    for i in range(len(A)):
        if A[i][i] == 0:
            return ("Main diagonal should not have zero!")

    xk0 = zeromatrix(len(A), 1)
    xk1 = zeromatrix(len(A), 1)
    for i in range(len(xk0[0])):
        for j in range(len(xk0)):
            xk0[j][i] = 1

    # print("xk1",xk1)
    c = 0
    def inf_norm(x,y):
        s = 0
        for i in range(len(A)):
            s+= x[i]*y[i]
        return s
    while inf_norm(xk1, xk0) >= eps:

        if c != 0:
            for i in range(len(xk1)):
                for j in range(len(xk1[i])):
                    xk0[i][j] = xk1[i][j]
        for i in range(len(A)):
            sum1 = 0
            sum2 = 0
            for j in range(i + 1, len(A[i])):
                sum2 = sum2 + (A[i][j] * xk0[j][0])
            for j in range(0, i):
                sum1 = sum1 + (A[i][j] * xk1[j][0])
            xk1[i][0] = (1 / A[i][i]) * (B[i][0] - sum1 - sum2)

        c = c + 1
    # print("c=",c)

    return xk1
def ConjuGra(A,b,e):
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A,x)
    d = r.copy()
    i = 1
    for i in range(n):
        while np.dot(r,r) > e:
            u = np.dot(A,d)
            al = np.dot(d,r)/np.dot(d,u)
            x = x + al*d
            r = b - np.dot(A,x)
            be = -np.dot(d,r)/np.dot(d,u)
            d = r + be*d
    return x
#Givans method
def crossprod(A,B):
    if len(A[0]) == len(B):
        crossprod = [[0 for i in range(len(B[0]))]for j in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for m in range(len(A)):
                    crossprod[i][j] = crossprod[i][j] + A[i][m]*B[m][j]
        return crossprod
    else:
        print("Matrices cannot be multiplied")
# crossprod is used in the function gaussgivan
def maxoff(A):
    maxtemp = A[0][1]
    k = 0
    l = 1
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            if abs(A[i][j]) > abs(maxtemp):
                maxtemp = A[i][j]
                k = i
                l = j
    return maxtemp, k, l


def gaussgivan(A, ep):
    max, i, j = maxoff(A)
    while abs(max) >= ep:
        #calculating theta
        if A[i][i] - A[j][j] == 0:
            theta = math.pi / 4
        else:
            theta = math.atan((2 * A[i][j]) / (A[i][i] - A[j][j])) / 2
        #Identity matrix
        P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        #Making P matix tridiagonal
        P[i][i] = P[j][j] = math.cos(theta)
        P[i][j] = -1 * math.sin(theta)
        P[j][i] = math.sin(theta)
        AP = crossprod(A, P)
        #making P an array so to use transpose function
        P = np.array(P)
        #Transpose of P
        PT = P.T.tolist()
        #getting back the matrix in tridiagonal form
        A = crossprod(PT, AP)
        #checking the offset in the matrix obtained
        max, i, j = maxoff(A)
    return A
#Power method
import math
#frobenius norm
def frob_norm(A):
    sum = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            sum = sum + (A[i][j] ** 2)
    return math.sqrt(sum)
#gives the norm of A
def pow_norm(A):
    max = 0
    for i in range(len(A)):
        if max <= A[i][0]:
            max = A[i][0]
    normA = scaler_matrix_division(max, A)
    return normA


def pow_method(A, x0=[[1], [1], [1]], eps=1.0e-4):
    i = 0
    lam0 = 1
    lam1 = 0
    while abs(lam1 - lam0) >= eps:
        # print("error=",abs(lam1-lam0))
        if i != 0:
            lam0 = lam1

        Ax0 = mat_mult(A, x0)
        AAx0 = mat_mult(A, Ax0)
        # print("Ax0=",Ax0)
        # print("AAx0=",AAx0)
        dotU = inner_product(AAx0, Ax0)
        dotL = inner_product(Ax0, Ax0)
        # print("U=",dotU)
        # print("L=",dotL)
        lam1 = dotU / dotL

        x0 = Ax0
        i = i + 1
        # print("i=",i)

        # print("eigenvalue=",lam1)
        ev = pow_norm(x0)
        # print ("eigenvector=",ev)
    return lam1, ev  # returns lam1=largest eigen value and ev = coressponding eigen vec
#gives mean
def Mean(A):
    n = len(A)
    sum = 0
    mean = 0
    for i in range(n):
        sum = sum + A[i]
    return sum/n
#gives variance
def Variance(A):
    n = len(A)
    mean = Mean(A)
    sum = 0
    for i in range(n):
        sum = sum + (A[i]-mean)**2
    return sum/n
#solves equation
def solveeqn(m, qw):
    m = Invert(m)

    X = []
    X.append(m[0][0]*qw[0] + m[0][1]*qw[1])
    X.append(m[1][0]*qw[0] + m[1][1]*qw[1])
    return(X)
def sum1(X, n):
    n = n + 1
    suMatrix = []
    j = 0
    while j<2*n:
        sum = 0
        i = 0
        while i< len(X):
            sum = sum + (X[i])**j
            i = i + 1
        suMatrix.append(sum)
        j = j+1
    return suMatrix
#makes a new matrix
def makemat(suMatrix, n):
    n = n + 1
    m = [[0 for i in range(n)]for j in range(n)]
    i = 0
    while i<n:
        j = 0
        while j<n:
            m[i][j] = suMatrix[j+i]
            j = j+1
        i = i + 1
    return m

def sum2(X, Y, n):
    n = n+1
    suMatrix = []
    j = 0
    while j<n:
        sum = 0
        i = 0
        while i< len(X):
            sum = sum + ((X[i])**j)*Y[i]
            i = i + 1
        suMatrix.append(sum)
        j = j+1
    return suMatrix

#chi square fit function
def fit(X,Y):
    k = sum1(X, 1)         #taking all the sigma_x
    m = makemat(k, 1)      #sigma_x**i matrix

    qw = sum2(X, Y, 1)     #sigma_x**i*y matrix

    X = solveeqn(m, qw)
    return X[0],X[1]

# Bootstrap method
def bootstrap(A,b):
    mean = []
    vari = []
    for i in range(b):
        #making bootstrap dataset
        resample = random.choices(A,k=len(A))
        #calculating mean of the resampled data
        m = Mean(resample)
        mean.append(m)
        var = Variance(resample)
        vari.append(var)
    #to get confidence levels we calculate Standard deviation of this distribution
    x = (Mean(mean))
    y = (Mean(var))
    #plotting the mean values as a histogram
    plt.hist(mean)
    return x,y

#jackknife method
def jkknife(A):
    n = len(A)
    yi = []
    for i in range(n):
        B = A.copy()
        del(B[i])
        #calculating mean excluding one element
        mean = Mean(B)
        #MAking a new y vector, stores all means
        yi.append(mean)
    #mean of the new formed set
    yibar = Mean(yi)
    sum = 0
    for i in range(n):
        sum = sum + (yi[i] - yibar)**2
    #calculating error
    err = ((n-1)/n)*sum
    return yibar,err
















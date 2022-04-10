import numpy as np
import sys
import math
import copy
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
#creating augmented matrix
def a_mat(A,B):
    #creation of zero mat with required rows and column numbers
    a_AB = [[0 for a in range(len(A))]for b in range(len(A)+1)]
    for i in range(len(A)):
        for j in range(len(A)+1):
            if j>=(len(A)):
                a_AB[j][i] = B[i]
            else :
                a_AB[j][i] = A[j][i]
    return a_AB

#creating reduced Row Echelon Form
def GauJo(a_mat):
    for i in range(len(a_mat[0])):
        p = a_mat[i][i]
        for j in range(len(a_mat)):
            a_mat[j][i] = a_mat[j][i]/p
        for k in range(len(a_mat[0])):
            if k == i or a_mat[i][k] == 0:
                next
            else:
                factor = a_mat[i][k]
                for l in range(len(a_mat)):
                    a_mat[l][k] = a_mat[l][k] - factor*a_mat[l][i]

    return a_mat[len(a_mat)-1]

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
    return x

#tolerance level already added here
# Jacobi Method
def Jacobi_Inv(A,B, x=None, tol = 10**(-4)):
    res = []
    # Initial guess if required
    if x is None:
        x = np.zeros(len(A))

    # vector of the diagonal elements of A and subtract

    D = np.diag(A)
    LU = A - np.diagflat(D)

    # Iterate till tolerance
    err = np.inf
    k = 0
    re = 0
    while err>tol:
        # Storing previous x assumed
        x_p = copy.deepcopy(x)
        x = (B - np.dot(LU,x)) / D
        x_p = x-x_p
        err = sum(x_p[i]**2 for i in range(len(x)))
        k+=1
        res.append(math.sqrt(err))
    return x,res,k

#Gauss Seidel
#tolerance level is added
def Gau_Seid(A, B, x=None, tol=1e-5):
    n = len(A)
    res = []
    k = 0
    if x is None: x = np.zeros(n)
    err = np.inf

    while err > tol:
        sum = 0
        # calculation of x
        for i in range(n):
            d = B[i]
            for j in range(n):
                if (i != j):
                    d -= A[i][j] * x[j]
            # Storing previous and updating the value of our solution
            temp = x[i]
            x[i] = d / A[i][i]
            sum += (x[i] - temp) ** 2

        # Error update
        err = sum
        res.append(math.sqrt(err))
        k+=1
    return x,res,k

def ConjuGra(A,b,e):
    n = len(b)
    res = []
    k = 0
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
            res.append(r)
            k+=1
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

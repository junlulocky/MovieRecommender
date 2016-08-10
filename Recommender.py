"""
# Title: Movie Recommender System Based on Matrix Factorization
# Description: train and test on the movielens100k.mat dataset
               The dataset for this project comes from MovieLens (http://grouplens.org/).
                It consists of 100,000 ratings from 1000 users on 1700 movies.
                The rating values go from 0 to 5. The dataset is split into train
                and test data, around 95,000 and 5,000 ratings respectively.

"""

print __doc__

import scipy.io
import numpy as np


# load data
mat = scipy.io.loadmat('movielens100k.mat')
R = mat['ratings']
R = np.array(R.todense().T)
U, M = R.shape  # U is the number of users, M is the number of movies

def data_preprocessing():
    # generating test data and train data
    nM_perU = np.array([len(np.nonzero(R[k])[0]) for k in range(U)]) # numbers of movies per user
    nU_perM = np.array([len(np.nonzero(R[:, k])[0]) for k in range(M)]) # numbers of users who rated a movie
    numTe = 40
    idxM = np.where(nU_perM > 10)[0]
    idxU = np.where(nM_perU > 40)[0]

    dd = np.array([])
    nn = np.array([])
    x = np.array([])

    for u, idxu in enumerate(idxU):
        idxOb = np.nonzero(R[idxu, :])[0]
        perm = np.random.permutation(range(len(idxOb)))
        idxRd = perm[0:numTe] # ??
        d = idxOb[idxRd]
        d = np.intersect1d(d, idxM)   # choose the data in which, the users rated more than 40 movies, and the movies are rated by more than 10 users
        dd = np.concatenate((dd, d))  # keep all the movie indexes for all users
        nn = np.concatenate((nn, u*np.ones(len(d)))) # keep the user indexes
        x = np.concatenate((x, R[u, d]))

    # split the raw samples into train and test sets
    RTe = np.zeros(shape=[U, M])
    RTe[nn.astype(int), dd.astype(int)] = x # test set
    R[nn.astype(int), dd.astype(int)] = 0 # train set

    return RTe, R

def matrix_factorize_ALS(X, hid_dim, lamb=0.08, max_iter=100, isSparse=False, Criterion=1e-5):
    """

    :param X: The matrix to be factorized
    :param hid_dim: hidden dimension
    :param lamb: tradeoff of penalization
    :param max_iter: max iteration
    :param isSparse: indicator whether the training matrix is sparse or not
    :param Criterion: not used now,
    :return: factorized matrix, and reconstructed matrix
    """
    num_row, num_col = X.shape
    if isSparse:
        ## get all the non zeros rows and columns
        idxRow = [np.nonzero(X[k])[0] for k in range(num_row)]
        idxColumn = [np.nonzero(X[:, k])[0] for k in range(num_col)]

        ## get the nonzero indexes
        idxTr = np.nonzero(X)
        numTotalTr = len(idxTr[0])  # numbers of nonzero entries in train datas
    else:
        ## get all the non zeros rows and columns
        idxRow = [range(len(X[k])) for k in range(num_row)]
        idxColumn = [range(len(X[:, k])) for k in range(num_col)]

        ## get all the indexes
        idxTr = np.where((X==0) | (X!=0))
        numTotalTr = len(idxTr[0])  # numbers of nonzero entries in train datas

    hid_dim += 2 # add two columns for the user and moview biases
    P = np.random.rand(num_row, hid_dim)
    Q = np.random.rand(num_col, hid_dim)
    P[:, hid_dim-1] = np.ones(num_row) # assume the first column is the bias of each user
    Q[:, 0] = np.ones(num_col)  # assum the last column is the bias of each film
    L = np.zeros(max_iter)

    # non-negative
    # P = abs(P)
    # Q = abs(Q)

    for k in range(max_iter):
        # ALS update
        for m in range(num_col):
            idx = idxColumn[m] # indexes of nonzero entries of movie m
            nm = len(idx)
            pm = P[idx, 1:hid_dim]
            rm = X[idx, m]
            tmp = np.linalg.inv(pm.T.dot(pm) + lamb*nm*np.eye(hid_dim-1))
            Q[m, 1:hid_dim] = (tmp.dot(pm.T).dot(rm))  # do not update the first column, it's always 1's for the bias of each user
            # non-negative
            # Q[m, 1:F] = Q[m, 1:F]*(Q[m, 1:F] >= 0)
        for u in range(num_row):
            idx = idxRow[u] # indexes of nonzero entries of user u
            nu = len(idx)
            qu = Q[idx, 0:hid_dim-1]
            ru = X[u, idx]
            tmp = np.linalg.inv(qu.T.dot(qu) + lamb*nu*np.eye(hid_dim-1))
            P[u, 0:hid_dim-1] = (tmp.dot(qu.T).dot(ru)) # do not update the las column, it's always 1's for the bias of each film
            # non-negative
            # P[u, 0:F-1] = P[u, 0:F-1]*(P[u, 0:F-1] >= 0)
        R_pred = P.dot(Q.T)
        R_pred = R_pred + np.sum(X[idxTr]-R_pred[idxTr])/numTotalTr
        L[k] = np.sum((X[idxTr]-R_pred[idxTr])**2)/numTotalTr
        print(('iteration %i: Train error=%.15f') %(k, L[k]))
        # if k > 0 and L[k-1] - L[k] < Criterion:
        #     break
    return  P, Q, R_pred


def test_recommender():
    R_test, R_Train = data_preprocessing() # shape=(num_user, num_movie)

    idxTr = np.nonzero(R_Train)
    numTotalTr = len(idxTr[0])  # numbers of nonzero entries in test datas
    idxTe = np.nonzero(R_test)
    numTotalTe = len(idxTe[0]) # numbers of nonzero entries in test datas

    # initialize
    lamb = 0.08  # the value of lambda can be set by cross validation
    F = 10     # choose the dimension of feature by cross validation
    maxIter = 30 # try different iteration
    Criterion = 1e-5

    ## Matrix factorization
    P, Q, R_pred = matrix_factorize_ALS(R_Train, F, lamb=lamb, max_iter=maxIter, isSparse=True, Criterion=Criterion)
    R_pred = P.dot(Q.T)

    R_pred[np.where(R_pred > 5)] = 5
    R_pred[np.where(R_pred < 0)] = 0

    RMSETr = np.sum((R_Train[idxTr]-R_pred[idxTr])**2) / numTotalTr
    print('Train error', RMSETr)

    RMSETe = np.sum((R_test[idxTe]-R_pred[idxTe])**2) / numTotalTe
    print('Test error', RMSETe)

if __name__ == '__main__':
    test_recommender()


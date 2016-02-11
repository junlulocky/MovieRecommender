"""
# Title: Movie Recommender System Based on Matrix Factorization
# Author: Jun Lu
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


def test_recommender():
    R_test, R_Train = data_preprocessing()

    idxTr = np.nonzero(R_Train)
    numTotalTr = len(idxTr[0]) # numbers of nonzero entries in train datas
    idxTe = np.nonzero(R_test)
    numTotalTe = len(idxTe[0]) # numbers of nonzero entries in test datas

    # save the indices of zero elements
    idxRow = [np.nonzero(R_Train[k])[0] for k in range(U)]
    idxColumn = [np.nonzero(R_Train[:, k])[0] for k in range(M)]


    # take log
    # R[idxTr] = np.log(R[idxTr])

    # initialize
    lamb = 0.08  # the value of lambda can be set by cross validation
    F = 10     # choose the dimension of feature by cross validation
    maxIter = 30 # try different iteration
    Criterion = 1e-5

    F += 2 # add two columns for the user and moview biases
    P = np.random.rand(U, F)
    Q = np.random.rand(M, F)
    P[:, F-1] = np.ones(U) # assume the first column is the bias of each user
    Q[:, 0] = np.ones(M)  # assum the last column is the bias of each film
    L = np.zeros(maxIter)

    # non-negative
    # P = abs(P)
    # Q = abs(Q)

    for k in range(maxIter):
        # ALS update
        for m in range(M):
            idx = idxColumn[m] # indexes of nonzero entries of movie m
            nm = len(idx)
            pm = P[idx, 1:F]
            rm = R_Train[idx, m]
            tmp = np.linalg.inv(pm.T.dot(pm) + lamb*nm*np.eye(F-1))
            Q[m, 1:F] = (tmp.dot(pm.T).dot(rm))  # do not update the first column, it's always 1's for the bias of each user
            # non-negative
            # Q[m, 1:F] = Q[m, 1:F]*(Q[m, 1:F] >= 0)
        for u in range(U):
            idx = idxRow[u] # indexes of nonzero entries of user u
            nu = len(idx)
            qu = Q[idx, 0:F-1]
            ru = R_Train[u, idx]
            tmp = np.linalg.inv(qu.T.dot(qu) + lamb*nu*np.eye(F-1))
            P[u, 0:F-1] = (tmp.dot(qu.T).dot(ru)) # do not update the las column, it's always 1's for the bias of each film
            # non-negative
            # P[u, 0:F-1] = P[u, 0:F-1]*(P[u, 0:F-1] >= 0)
        R_pred = P.dot(Q.T)
        R_pred = R_pred + np.sum(R_Train[idxTr]-R_pred[idxTr])/numTotalTr
        L[k] = np.sum((R[idxTr]-R_pred[idxTr])**2)/numTotalTr
        print(('iteration %i: Train error=%.15f') %(k, L[k]))
        if k > 0 and L[k-1] - L[k] < Criterion:
            break

    # R_pred = np.exp(R_pred)

    R_pred[np.where(R_pred > 5)] = 5
    R_pred[np.where(R_pred < 0)] = 0

    RMSETe = np.sum((R_test[idxTe]-R_pred[idxTe])**2) / numTotalTe
    print('Test error', RMSETe)

if __name__ == '__main__':
    test_recommender()


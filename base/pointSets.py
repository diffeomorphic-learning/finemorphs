import logging
import numpy as np
from numba import jit
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA


__s0Coeff__ = 0.
@jit(nopython=True)
def _RidgeSumOfSquares(u):
    return (u[1:, :] ** 2).sum()
@jit(nopython=True)
def _RidgeSumOfSquaresGradient(u):
    return np.concatenate((np.zeros((1, u.shape[1])), 2*u[1:, :]), axis=0)
@jit(nopython=True)
def _RidgeSumOfSquaresToSim(u):
    d = np.diag(u[1:,:])
    return (u[1:, :]**2).sum() - (d.sum())**2/d.shape[0]
@jit(nopython=True)
def _RidgeSumOfSquaresToSimGradient(u):
    d = np.diag(u[1:,:])
    g = 2 * u[1:, :] - 2 * (d.sum()/d.shape[0]) * np.eye((u.shape[0]-1, u.shape[1]))
    return np.concatenate((np.zeros((1, u.shape[1])), g), axis=0)
@jit(nopython=True)
def _RidgeSumOfSquaresToEye(u):
    return ((u[1:, :] - np.eye(u.shape[0]-1, u.shape[1])) ** 2).sum()
@jit(nopython=True)
def _RidgeSumOfSquaresToEyeGradient(u):
    return np.concatenate((np.zeros((1, u.shape[1])), 2*(u[1:, :] - np.eye(u.shape[0]-1, u.shape[1]))), axis=0)

@jit(nopython=True)
def _ridgecost(u):
    return _RidgeSumOfSquaresToSim(u)
@jit(nopython=True)
def _ridgecostgradient(u):
    return _RidgeSumOfSquaresToSimGradient(u)


def LogisticScore(x, y, w=None, sigmaError=1.0):
    if w is None:
        w = np.ones((x.shape[0], 1))
    res = (np.ravel(w) * (- x[np.arange(x.shape[0])[:, np.newaxis], y].sum(axis=1) + logsumexp(x, axis=1))).sum()
    res = res / (sigmaError ** 2)
    return res


def LogisticScoreGradient(x, y, w=None, sigmaError=1.0):    # called by getGradient
                                                            # returns Del_finalState of LogisticScore
    if w is None:
        w = np.ones((x.shape[0],1))
    gu = np.exp(x)
    pu = gu/gu.sum(axis=1)[:,np.newaxis]
    ni = np.eye(x.shape[1])
    m = np.dot(pu, ni)  # = pu
    grad = (-ni[np.arange(ni.shape[0]), y] + m)*w
    grad = grad / (sigmaError ** 2)
    return grad


def RegressionScore(x, y, sigmaError=1.0):
    res = ((y - x[:,:y.shape[1]]) ** 2).sum()
    res = res / (sigmaError ** 2)
    return res


def RegressionScoreGradient(x, y, sigmaError=1.0):  # called by getGradient
                                                    # returns Del_finalState of RegressionScore
    grad = np.zeros(x.shape)
    grad[:, :y.shape[1]] = -2 * (y - x[:, 0:y.shape[1]])
    grad = grad / (sigmaError ** 2)
    return grad


def RidgeRegularization(u, lam=(0.0,), ridgecost=_ridgecost):
    res = 0.0
    nA = len(lam)
    for layer_A in range(nA):
        res += lam[layer_A] * ridgecost[layer_A](u[layer_A])
    return res


def RidgeGradientInU(u, lam=(0.0,), ridgecostgradient=_ridgecostgradient):
                                                                    # called by getGradient
                                                                    # returns Del_u of ridge regularization term
    grad = [None] * len(lam)
    nA = len(lam)
    for layer_A in range(nA):
        grad[layer_A] = lam[layer_A] * ridgecostgradient[layer_A](u[layer_A])
    return grad


def build3DProjection(X, test=None, Y=None, mode='classification'): 
    xTr3 = None
    xTe3 = None
    # if X.shape[1] < 3:
    #     xTr3 = np.concatenate((X, np.zeros((X.shape[0], 3-X.shape[1]))), axis = 1)
    #     if test is not None:
    #         xTe3 = np.concatenate((test, np.zeros((test.shape[0], 3 - test.shape[1]))), axis=1)
    # elif X.shape[1] == 3:
    #     xTr3 = np.copy(X)
    #     if test is not None:
    #         xTe3 = np.copy(test)
    # else:   # X.shape[1] > 3
    if Y is None:
        pca = PCA(3)
        xTr3 = pca.fit_transform(X)
        if test is not None:
            xTe3 = pca.transform(test)
    else:
        if mode == 'classification':
            cl = np.unique(Y)
            nc = min(len(cl)-1, 3)
            lda = LDA(n_components=nc)
            xTr3 = lda.fit_transform(X, Y[:,0])
            if test is not None:
                xTe3 = lda.transform(test)
        elif mode == 'regression':
            nc = min(Y.shape[1], 3)
            cca = CCA(n_components=nc)
            xTr3 = cca.fit_transform(X, Y)[0]
            if test is not None:
                xTe3 = cca.transform(test)
        else:
            logging.error('Unknown prodiction mode in build3Dprojection')
            return
        if nc < 3: 
            lin = LinearRegression()
            lin.fit(xTr3, X)
            res = X - lin.predict(xTr3)
            # print('nc = ', nc)
            # print('res.shape = ', res.shape)
            pca = PCA(3 - nc)
            res_ = pca.fit_transform(res)
            xTr3 = np.concatenate((xTr3, res_), axis=1)
            if test is not None:
                res2 = test - lin.predict(xTe3)
                res2_ = pca.transform(res2)
                xTe3 = np.concatenate((xTe3, res2_), axis=1)

    if test is None:
        return xTr3
    else:
        return xTr3, xTe3


def runPCA(X):
    pca = PCA()
    pca.fit_transform(X)
    return np.sqrt(pca.singular_values_)


def read3DVector(filename):
    try:
        with open(filename, 'r') as fn:
            ln0 = fn.readline()
            N = int(ln0[0])
            v = np.zeros([N, 3])
            for i in range(N):
                ln0 = fn.readline().split()
                for k in range(3):
                    v[i, k] = float(ln0[k])
    except IOError:
        print('cannot open ', filename)
        raise
    return v


def loadlmk(filename, dim=3):
    # [x, label] = loadlmk(filename, dim)
    # Loads 3D landmarks from filename in .lmk format.
    # Determines format version from first line in file
    #   if version number indicates scaling and centering, transform coordinates...
    # the optional parameter s in a 3D scaling factor

    try:
        with open(filename, 'r') as fn:
            ln0 = fn.readline()
            versionNum = 1
            versionStrs = ln0.split("-")
            if len(versionStrs) == 2:
                try:
                    versionNum = int(float(versionStrs[1]))
                except:
                    pass

            ln = fn.readline().split()
            N = int(ln[0])
            x = np.zeros([N, dim])
            label = []

            for i in range(N):
                ln = fn.readline()
                label.append(ln)
                ln0 = fn.readline().split()
                for k in range(dim):
                    x[i, k] = float(ln0[k])
            if versionNum >= 6:
                lastLine = ''
                nextToLastLine = ''
                # read the rest of the file
                # the last two lines contain the center and the scale variables
                while 1:
                    thisLine = fn.readline()
                    if not thisLine:
                        break
                    nextToLastLine = lastLine
                    lastLine = thisLine

                centers = nextToLastLine.rstrip('\r\n').split(',')
                scales = lastLine.rstrip('\r\n').split(',')
                if len(scales) == dim and len(centers) == dim:
                    if scales[0].isdigit and scales[1].isdigit and scales[2].isdigit and centers[0].isdigit \
                            and centers[1].isdigit and centers[2].isdigit:
                        x[:, 0] = x[:, 0] * float(scales[0]) + float(centers[0])
                        x[:, 1] = x[:, 1] * float(scales[1]) + float(centers[1])
                        x[:, 2] = x[:, 2] * float(scales[2]) + float(centers[2])

    except IOError:
        print('cannot open ', filename)
        raise
    return x, label


def savelmk(x, filename):
    # savelmk(x, filename)
    # save landmarks in .lmk format.

    with open(filename, 'w') as fn:
        str = 'Landmarks-1.0\n {0: d}\n'.format(x.shape[0])
        fn.write(str)
        for i in range(x.shape[0]):
            str = '"L-{0:d}"\n'.format(i)
            fn.write(str)
            str = ''
            for k in range(x.shape[1]):
                str = str + '{0: f} '.format(x[i, k])
            str = str + '\n'
            fn.write(str)
        fn.write('1 1 \n')


# Reads in .vtk format
def readPoints(fileName):
    with open(fileName, 'r') as fvtkin:
        fvtkin.readline()
        fvtkin.readline()
        fvtkin.readline()
        fvtkin.readline()
        ln = fvtkin.readline().split()
        npt = int(ln[1])
        x = np.array((npt, 3))
        for ll in range(x.shape[0]):
            ln = fvtkin.readline().split()
            x[ll, 0] = float(ln[0])
            x[ll, 1] = float(ln[1])
            x[ll, 2] = float(ln[2])
    return x


# Saves in .vtk format
def savePoints(fileName, x, vector=None, scalars=None):
    if x.shape[1] < 3:
        x = np.concatenate((x, np.zeros((x.shape[0], 3 - x.shape[1]))), axis=1)
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET UNSTRUCTURED_GRID\n')
        fvtkout.write('\nPOINTS {0: d} float'.format(x.shape[0]))
        for ll in range(x.shape[0]):
            fvtkout.write('\n{0: f} {1: f} {2: f}'.format(x[ll, 0], x[ll, 1], x[ll, 2]))
        if vector is None and scalars is None:
            return
        fvtkout.write(('\nPOINT_DATA {0: d}').format(x.shape[0]))
        if scalars is not None:
            fvtkout.write('\nSCALARS scalars float 1\nLOOKUP_TABLE default')
            for ll in range(x.shape[0]):
                fvtkout.write('\n {0: .5f} '.format(scalars[ll]))

        if vector is not None:
            fvtkout.write('\nVECTORS vector float')
            for ll in range(x.shape[0]):
                fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(vector[ll, 0], vector[ll, 1], vector[ll, 2]))

        fvtkout.write('\n')


# Saves in .vtk format
def saveTrajectories(fileName, xt):
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\ncurves \nASCII\nDATASET POLYDATA\n')
        fvtkout.write('\nPOINTS {0: d} float'.format(xt.shape[0] * xt.shape[1]))
        if xt.shape[2] == 2:
            xt = np.concatenate(xt, np.zeros([xt.shape[0], xt.shape[1], 1]))
        for t in range(xt.shape[0]):
            for ll in range(xt.shape[1]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(xt[t, ll, 0], xt[t, ll, 1], xt[t, ll, 2]))
        nlines = (xt.shape[0] - 1) * xt.shape[1]
        fvtkout.write('\nLINES {0:d} {1:d}'.format(nlines, 3 * nlines))
        for t in range(xt.shape[0] - 1):
            for ll in range(xt.shape[1]):
                fvtkout.write('\n2 {0: d} {1: d}'.format(t * xt.shape[1] + ll, (t + 1) * xt.shape[1] + ll))
        fvtkout.write('\n')



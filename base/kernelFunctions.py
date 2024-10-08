from numba import jit
import numpy as np
from math import pi, exp, sqrt
from . import kernelFunctions_util as ku
from scipy.spatial import distance as dfun


def kernelMatrixGauss(x, firstVar=None, grid=None, par=[1], diff = False, diff2 = False, constant_plane=False, precomp=None):
    sig = par[0]
    sig2 = 2*sig*sig
    if precomp is None:
        if firstVar is None:
            if grid is None:
                u = np.exp(-dfun.pdist(x,'sqeuclidean')/sig2)
                K = dfun.squareform(u, checks=False)
                np.fill_diagonal(K, 1)
                precomp = np.copy(K)
                if diff:
                    K = -K/sig2
                elif diff2:
                    K = K/(sig2*sig2)
            else:
                dst = ((grid[..., np.newaxis, :] - x)**2).sum(axis=-1)
                K = np.exp(-dst/sig2)
                if diff:
                    K = -K/sig2
                elif diff2:
                    K = K/(sig2*sig2)
        else:
            K = np.exp(-dfun.cdist(firstVar, x, 'sqeuclidean')/sig2)
            precomp = np.copy(K)
            if diff:
                K = -K/sig2
            elif diff2:
                K = K/(sig2*sig2)
    else:
        K = np.copy(precomp)
        if diff:
            K = -K/sig2
        elif diff2:
            K = K/(sig2*sig2)

    if constant_plane:
        K2 = np.exp(-dfun.pdist(x[:,x.shape[1]-1],'sqeuclidean')/sig2)
        np.fill_diagonal(K2, 1)
        if diff:
            K2 = -K2/sig2
        elif diff2:
            K2 = K2/(sig2*sig2)
        return K, K2, precomp
    else:
        return K, precomp


# Polynomial factor for Laplacian kernel
@jit(nopython=True)
def lapPol(u, ord):
    if ord == 0:
        pol = 1.
    elif ord == 1:
        pol = 1. + u
    elif ord == 2:
        pol = (3. + 3*u + u*u)/3.
    elif ord == 3:
        pol = (15. + 15 * u + 6*u*u + u*u*u)/15.
    else:
        pol = (105. + 105*u + 45*u*u + 10*u*u*u + u*u*u*u)/105.
    return pol


# Polynomial factor for Laplacian kernel (first derivative)
@jit(nopython=True)
def lapPolDiff(u, ord):
    if ord == 1:
        pol = 1.
    elif ord == 2:
        pol = (1 + u)/3.
    elif ord == 3:
        pol = (3 + 3*u + u*u)/15.
    else:
        pol = (15 + 15 * u + 6*u*u + u*u*u)/105.
    return pol


# Polynomial factor for Laplacian kernel (second derivative)
@jit(nopython=True)
def lapPolDiff2(u, ord):
    pol = 0
    if ord == 2:
        pol = 1.0/3.
    elif ord == 3:
        pol = (1 + u)/15.
    else:
        pol = (3 + 3 * u + u*u)/105.
    return pol


def kernelMatrixLaplacian(x, firstVar=None, grid=None, par=(1., 3), diff=False, diff2 = False, constant_plane=False, precomp = None):
    sig = par[0]
    ord=par[1]
    if precomp is None:
        precomp = kernelMatrixLaplacianPrecompute(x, firstVar, grid, par)

    u = precomp[0]
    expu = precomp[1]

    if firstVar is None and grid is None:
        if diff==False and diff2==False:
            K = dfun.squareform(lapPol(u,ord) *expu)
            np.fill_diagonal(K, 1)
        elif diff2==False:
            K = dfun.squareform(-lapPolDiff(u, ord) * expu/(2*sig*sig))
            np.fill_diagonal(K, -1./((2*ord-1)*2*sig*sig))
        else:
            K = dfun.squareform(lapPolDiff2(u, ord) *expu /(4*sig**4))
            np.fill_diagonal(K, 1./((35)*4*sig**4))
    else:
        if diff==False and diff2==False:
            K = lapPol(u,ord) * expu
        elif diff2==False:
            K = -lapPolDiff(u, ord) * expu/(2*sig*sig)
        else:
            K = lapPolDiff2(u, ord) *expu/(4*sig**4)

    if constant_plane:
        uu = dfun.pdist(x[:,x.shape[1]-1])/sig
        K2 = dfun.squareform(lapPol(uu,ord)*np.exp(-uu))
        np.fill_diagonal(K2, 1)
        return K,K2,precomp
    else:
        return K,precomp

def kernelMatrixLaplacianPrecompute(x, firstVar=None, grid=None, par=(1., 3), diff=False, diff2 = False, constant_plane=False):
    sig = par[0]
    ord=par[1]
    if firstVar is None:
        if grid is None:
            u = dfun.pdist(x)/sig
        else:
            u = np.sqrt(((grid[..., np.newaxis, :] - x)**2).sum(axis=-1))/sig
    else:
        u = dfun.cdist(firstVar, x)/sig
    precomp = [u, np.exp(-u)]
    return precomp

# Wrapper for kernel matrix computation
def  kernelMatrix(Kpar, x, firstVar=None, grid=None, diff = False, diff2=False, constant_plane = False):
    # [K, K2] = kernelMatrix(Kpar, varargin)
    # creates a kernel matrix based on kernel parameters Kpar
    # if varargin = z

    if Kpar._hold:
        precomp = np.copy(Kpar.precomp)
    else:
        precomp = None

    if Kpar.name == 'gauss':
        res = kernelMatrixGauss(x,firstVar=firstVar, grid=grid, par = [Kpar.sigma], diff=diff, diff2=diff2, constant_plane = constant_plane, precomp=precomp)
    elif Kpar.name == 'lap':
        res = kernelMatrixLaplacian(x,firstVar=firstVar, grid=grid, par = [Kpar.sigma, Kpar.order], diff=diff, diff2=diff2, constant_plane = constant_plane, precomp=precomp)
    else:
        print('unknown Kernel type')
        return []

    Kpar.precomp = res[-1]
    if constant_plane:
        return res[0:2]
    else:
        return res[0]


@jit(nopython=True)
def atanK(u):
    return np.arctan(u) + pi/2

@jit(nopython=True)
def atanKDiff(u):
    return 1/(1 + u**2)

@jit(nopython=True)
def logcoshK(u):
    v = np.fabs(u)
    return u + v + np.log1p(np.exp(-2*v))

@jit(nopython=True)
def logcoshKDiff(u):
    return 1 + np.tanh(u)

@jit(nopython=True)
def ReLUK(u):
    return np.maximum(u, 0)

@jit(nopython=True)
def ReLUKDiff(u):
    return heaviside(u)

@jit(nopython=True)
def heaviside(u):
    return (np.sign(u - 1e-8) + np.sign(u + 1e-8) + 2) / 4

@jit(nopython=True)
def applyK_(y, x, a, name, scale, order):
    res = np.zeros(a.shape)
    if name == 'min':
        for s in scale:
            u = np.minimum(y,x)/s
            res += ReLUK(u)*a
    elif name == 'gauss':
        for s in scale:
            res += np.exp(- ((y-x)**2).sum()/(2*s**2)) * a
    elif name == 'lap':
        for s in scale:
            u_ = 0.
            for j in range(y.size):
                u_ += (y[j] - x[j])**2
            u = sqrt(u_)/s
            res += lapPol(u, order) * np.exp(- u) *a
    return res /len(scale)

@jit(nopython=True)
def applyDiffKT_(y, x, a1a2, name, scale, order):
    res = np.zeros(y.shape)
    if name == 'min':
        for s in scale:
            u = np.minimum(y,x)/s
            res += (heaviside(x-y)*a1a2)*ReLUKDiff(u)/s
    elif name == 'gauss':
        for s in scale:
            res += (y-x) * (-np.exp(- ((y-x)**2).sum()/(2*s**2)) * (a1a2).sum())/(s**2)
    elif name == 'lap':
        for s in scale:
            u = np.sqrt(((y-x)**2).sum())/s
            res += (y-x) * (-lapPolDiff_(u, order) * np.exp(- u) * (a1a2).sum()/(s**2))
    return res /len(scale)


@jit(nopython=True)
def applyDiv_(y, x, a, name, scale, order):
    res = 0
    if name == 'min':
        for s in scale:
            u = np.minimum(y,x)/s
            res += (heaviside(x-y)*a*ReLUKDiff(u)/s).sum()
    elif name == 'gauss':
        for s in scale:
            res += ((y-x)*a).sum() * (-np.exp(- ((y-x)**2).sum()/(2*s**2)))/(s**2)
    elif name == 'lap':
        for s in scale:
            u = np.sqrt(((y-x)**2).sum())/s
            res += ((y-x)*a).sum() * (-lapPolDiff_(u, order) * np.exp(- u) /(s**2))
    return res /len(scale)



# Kernel specification
# name = 'gauss', 'lap', or 'min'
# affine = 'affine' or 'euclidean' (affine component)
# sigma: width
# order: order for Laplacian kernel
# w1: weight for linear part; w2: weight for translation part; center: origin
# dim: dimension
class KernelSpec:
    def __init__(self, name='gauss', affine ='none', sigma = (1.,), order = 10, w1 = 1.0,
                 w2=1.0, dim=3, center=None, weight=1.0, localMaps=None, pykeops_flag=False,
                 dtype='float64'):
        self.name = name
        self.pykeops_flag = pykeops_flag
        self.dtype = dtype
        if np.isscalar(sigma):
            self.sigma = np.array([sigma], dtype=float)
        else:
            self.sigma = np.array(sigma, dtype=float)
        self.order = order
        self.weight=weight
        self.w1 = w1
        self.w2 = w2
        self.constant_plane=False
        if center is None:
            self.center = np.zeros(dim)
        else:
            self.center = np.array(center)
        self.affine_basis = []
        self.dim = dim
        self.precomp = []
        self._hold = False
        self._state = False
        self.affine = affine
        self.localMaps = localMaps
        self.kff = False
        if name == 'lap':
            self.kernelMatrix = kernelMatrixLaplacian
            if self.order > 4:
                self.order = 3
            self.par = [sigma, self.order]
            self.kff = True
        elif name == 'gauss':
            self.kernelMatrix = kernelMatrixGauss
            self.order = 10 
            self.par = [sigma]
            self.kff = True
        elif name == 'min':
            self.kernelMatrix = None
            self.par = []
        else:
            self.name = 'none'
            self.kernelMatrix = None
            self.par = []
        if self.affine=='euclidean':
            if self.dim == 3:
                s2 = np.sqrt(2.0)
                self.affine_basis.append(np.mat([ [0,1,0], [-1,0,0], [0,0,0]])/s2)
                self.affine_basis.append(np.mat([ [0,0,1], [0,0,0], [-1,0,0]])/s2)
                self.affine_basis.append(np.mat([ [0,0,0], [0,0,1], [0,-1,0]])/s2)
            elif self.dim==2:
                s2 = np.sqrt(2.0)
                self.affine_basis.append(np.mat([ [0,1], [-1,0]])/s2)
            else:
                print('Euclidian kernels only available in dimensions 2 or 3')
                return


# Main class for kernel definition
class Kernel(KernelSpec):
    def precompute(self, x,  firstVar=None, grid=None, diff=False, diff2=False):
        if not (self.kernelMatrix is None):
            if self._hold:
                precomp = self.precomp
            else:
                precomp = None

            r = self.kernelMatrix(x, firstVar=firstVar, grid = grid, par = self.par, precomp=precomp, diff=diff, diff2=diff2)
            self.precomp = r[1]
            return r[0] * self.weight

    def hold(self):
        self._state = self._hold
        self._hold = True
    def release(self):
        self._state = False
        self._hold = False
    def reset(self):
        self._hold=self._state


    def getK(self, x, firstVar = None):
        z = None
        if not (self.kernelMatrix is None):
            if firstVar is None:
                z = ku.kernelmatrix(x, x, self.name, self.sigma, self.order)
            else:
                z = ku.kernelmatrix(x, firstVar, self.name, self.sigma, self.order)
        return z

    # Computes K(x,x)a or K(x,y)a
    def applyK(self, x, a, firstVar = None, grid=None,matrixWeights=False, dtype = None):
        if dtype is None:
            dtype = self.dtype
        if firstVar is None:
            y = np.copy(x)
            if matrixWeights:
                z = ku.applykmat(y, x, a, self.name, self.sigma, self.order)
            elif self.localMaps:
                if self.localMaps[2] == 'naive':
                    z = ku.applylocalk_naive(y ,x ,a ,self.name, self.sigma ,self.order)
                else:
                    z = ku.applylocalk(y ,x ,a ,self.name, self.sigma ,self.order, self.localMaps[0] ,
                                        self.localMaps[1])
            else:
                z = ku.applyK(y ,x ,a ,self.name, self.sigma, self.order, dtype=dtype,
                              pykeops_flag=self.pykeops_flag)
        else:
            if matrixWeights:
                z = ku.applykmat(firstVar, x, a, self.name, self.sigma, self.order)
            elif self.localMaps:
                if self.localMaps[2] == 'naive':
                    z = ku.applylocalk_naive(firstVar ,x ,a ,self.name, self.sigma ,self.order)
                else:
                    z = ku.applylocalk(firstVar ,x ,a ,self.name, self.sigma ,self.order , self.localMaps[0] ,
                                        self.localMaps[1])
            else:
                z = ku.applyK(firstVar ,x ,a , self.name, self.sigma ,self.order, dtype=dtype, pykeops_flag=self.pykeops_flag)
        if self.affine == 'affine':
            xx = x-self.center
            if firstVar is None:
                if grid is None:
                    z += self.w1 * np.dot(xx, np.dot(xx.T, a)) + self.w2 * a.sum(axis=0)
                else:
                    yy = grid -self.center
                    z += self.w1 * np.dot(grid, np.dot(xx.T, a)) + self.w2 * a.sum(axis=0)
            else:
                yy = firstVar-self.center
                z += self.w1 * np.dot(yy, np.dot(xx.T, a)) + self.w2 * a.sum(axis=0)
        elif self.affine == 'euclidean':
            xx = x-self.center
            if not (firstVar is None):
                yy = firstVar-self.center
            if not (grid is None):
                gg = grid - self.center
            z += self.w2 * a.sum(axis=0)
            for E in self.affine_basis:
                xE = np.dot(xx, E.T)
                if firstVar is None:
                    if grid is None:
                        z += self.w1 * (xE * a).sum() * xE
                    else:
                        gE = np.dot(gg, E.T)
                        z += self.w1 * (xE * a).sum() * gE
                else:
                    yE = np.dot(yy, E.T)
                    z += self.w1 * np.multiply(xE, a).sum() * yE
        return z

    # # Computes A(i) = sum_j D_1[K(x(i), x(j))a2(j)]a1(i)
    def applyDiffK(self, x, a1, a2):
        z = ku.applykdiff1(x, a1, a2, self.name, self.sigma, self.order)
        return z

    # Computes A(i) = sum_j D_2[K(x(i), x(j))a2(j)]a1(j)
    def applyDiffK2(self, x, a1, a2):
        z = ku.applykdiff2(x, a1, a2, self.name, self.sigma, self.order)
        return z

    def applyDiffK1and2(self, x, a1, a2):
        z = ku.applykdiff1and2(x, a1, a2, self.name, self.sigma, self.order)
        return z

    # Computes A(i) = sum_j D_2[K(x(i), x(j))a2(j)]a1(j)
    def applyDiffKmat(self, x, beta, firstVar=None):
        if firstVar is None:
            z = ku.applykdiffmat(x, x, beta, self.name, self.sigma, self.order)
        else:
            z = ku.applykdiffmat(firstVar, x, beta, self.name, self.sigma, self.order)
        return z

    # Computes array A(i) = sum_k sum_(j) nabla_1[a1(k,i). K(x(i), x(j))a2(k,j)]
    def applyDiffKT(self, x, p, a, firstVar=None, regWeight=1., lddmm=False, dtype=None):
        if dtype is None:
            dtype = self.dtype
        if firstVar is None:
            y = np.copy(x)
        else:
            y = firstVar
        if self.localMaps:
            if self.localMaps[2] == 'naive':
                zpx = ku.applylocalk_naivedifft(y ,x , p ,a , self.name, self.sigma ,self.order,
                                                regWeight=regWeight, lddmm=lddmm)
            else:
                zpx = ku.applylocalkdifft(y, x, p, a, self.name,  self.sigma, self.order, self.localMaps[0],
                                           self.localMaps[1], regWeight=regWeight, lddmm=lddmm)
        else:
            zpx = ku.applyDiffKT(y ,x , p ,a , self.name, self.sigma ,self.order,
                                 regWeight=regWeight, lddmm=lddmm, dtype=dtype, pykeops_flag=self.pykeops_flag)
        if self.affine == 'affine':
            xx = x-self.center

            ### TO CHECK
            zpx += self.w1 * ((a*xx).sum() * p + (p*xx).sum() * a)
        elif self.affine == 'euclidean':
            xx = x-self.center
            for E in self.affine_basis:
                yy = np.dot(xx, E.T)
                for k in range(len(a)):
                     bb = np.dot(p[k], E)
                     zpx += self.w1 * np.multiply(yy, a[k]).sum() * bb
        return zpx

    def testDiffKT(self, y, x, p, a):
        k0 = (p*self.applyK(x,a, firstVar=y)).sum()
        dy = np.random.randn(y.shape[0], y.shape[1])
        eps = 1e-8
        k1 = (p*self.applyK(x,a, firstVar=y+eps*dy)).sum()
        grd = self.applyDiffKT(x,p,a,firstVar=y)
        print('test diffKT: {0:.5f}, {1:.5f}'.format((k1-k0)/eps, (grd*dy).sum()))


    def applyDivergence(self, x,a, firstVar=None, dtype=None):
        if dtype is None:
            dtype = self.dtype
        if firstVar is None:
            y = x
        else:
            y = firstVar
        if self.localMaps:
            if self.localMaps[2] == 'naive':
                zpx = ku.applylocalk_naivediv(y, x,a, self.name, self.sigma, self.order)
            else:
                zpx = ku.applylocalkdiv(y, x, a, self.name, self.sigma, self.order, self.localMaps[0],
                                        self.localMaps[1])
        else:
            zpx = ku.applyDiv(y,x,a, self.name, self.sigma, self.order)
        return zpx


    def applyDDiffK11and12(self, x, n, a, p):
        z = ku.applykdiff11and12(x, n, a, p, self.name, self.sigma, self.order)
        return z


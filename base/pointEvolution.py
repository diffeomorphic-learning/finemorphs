import logging
import numpy as np
from . import affineBasis


class GaussDiff__:
    def eval(self, u):
        return u*np.exp(- (u**2)/2)
    def evalDiff(self, u):
        uu = u*u
        return (1-uu)*np.exp(- (uu)/2)


def landmarkDirectEvolutionEuler(x0, at, KparDiff, affine=None, withJacobian=False, withNormals=None,
                                 withPointSet=None, onlyPointSet=False, zt=None):
    dtype = KparDiff.dtype
    if not (affine is None or len(affine[0]) == 0):
        withaff = True
        A = affine[0]
        b = affine[1]
    else:
        withaff = False
        A = np.zeros((1,1,1))
        b = np.zeros((1,1))

    N = x0.shape[0]
    dim = x0.shape[1]
    T = at.shape[0]
    timeStep = 1.0 / (T)
    xt = np.zeros((T+1, N, dim))
    xt[0, ...] = x0

    Jt = np.zeros((T + 1, N, 1))
    simpleOutput = True
    if withPointSet is not None:
        simpleOutput = False
        K = withPointSet.shape[0]
        y0 = withPointSet
        yt = np.zeros((T + 1, K, dim))
        yt[0, ...] = y0
        if withJacobian:
            simpleOutput = False

    if withNormals is not None:
        simpleOutput = False
        nt = np.zeros((T+1, N, dim))
        nt[0, ...] = withNormals

    if withJacobian:
        simpleOutput = False

    if not onlyPointSet:
        for t in range(T):
            if withaff:
                Rk = affineBasis.getExponential(timeStep * A[t,:,:])
                xt[t+1,...] = np.dot(xt[t,...], Rk.T) + timeStep * b[t,:]
            else:
                xt[t+1, ...] = xt[t, ...]
            xt[t+1,...] += timeStep*KparDiff.applyK(xt[t,...], at[t,...], dtype=dtype)

            if withPointSet is not None:
                if withaff:
                    yt[t+1,...] = np.dot(yt[t, ...], Rk.T) + timeStep * b[t]
                else:
                    yt[t+1,...] = yt[t, ...]
                yt[t + 1, ...] += timeStep * KparDiff.applyK(xt[t, ...], at[t, ...], firstVar=yt[t, ...], dtype=dtype)
            if withJacobian:
                Jt[t + 1, ...] = Jt[t + 1, ...] + timeStep * KparDiff.applyDivergence(xt[t, :, :], at[t, :, :], dtype=dtype)
                if withaff:
                    Jt[t + 1, ...] += timeStep * (np.trace(A[t]))
            if withNormals is not None:
                nt[t + 1, ...] = nt[t, ...] - timeStep * KparDiff.applyDiffKT(xt[t, :, :], nt[t, :, :], at[t, :, :],
                                                                              dtype=dtype)
                if withaff:
                    nt[t + 1, :, :] -= timeStep * np.dot(nt[t, :, :], A[t])
    else:   # only propagate the PointSet
        if withPointSet is not None:
            if zt is not None:
                xt = np.copy(zt)
                for t in range(T):
                    if withaff:
                        Rk = affineBasis.getExponential(timeStep * A[t, :, :])
                        yt[t+1,...] = np.dot(yt[t, ...], Rk.T) + timeStep * b[t]
                    else:
                        yt[t+1,...] = yt[t, ...]
                    yt[t + 1, ...] += timeStep * KparDiff.applyK(xt[t, ...], at[t, ...], firstVar=yt[t, ...], dtype=dtype)
                    if withJacobian:
                        Jt[t + 1, ...] = Jt[t + 1, ...] + timeStep * KparDiff.applyDivergence(xt[t, :, :], at[t, :, :],
                                                                                              dtype=dtype)
                        if withaff:
                            Jt[t + 1, ...] += timeStep * (np.trace(A[t]))
                    if withNormals is not None:
                        nt[t + 1, ...] = nt[t, ...] - timeStep * KparDiff.applyDiffKT(xt[t, :, :], nt[t, :, :],
                                                                                      at[t, :, :], dtype=dtype)
                        if withaff:
                            nt[t + 1, :, :] -= timeStep * np.dot(nt[t, :, :], A[t])
            else:
                logging.info('\nerror:  onlyPointSet = True and zt = None')
                exit()
        else:
            logging.info('\nerror:  onlyPointSet = True and withPointSet = None')
            exit()

    if simpleOutput:
        return xt
    else:
        output = [xt]
        if not (withPointSet is None):
            output.append(yt)
        if not (withNormals is None):
            output.append(nt)
        if withJacobian:
            output.append(Jt)
        return output


def landmarkHamiltonianCovector(x0, at, px1, Kpardiff, regWeight, affine=None, zt=None, idx_diff=None, idx_nodiff=None):
    dtype = Kpardiff.dtype
    if not (affine is None or len(affine[0]) == 0):
        withaff = True
        A = affine[0]
    else:
        withaff = False
        A = np.zeros((1, 1, 1))

    N = x0.shape[0]
    dim = x0.shape[1]
    T = at.shape[0]
    timeStep = 1.0 / (T)

    pxt = np.zeros((T + 1, N, dim))
    pxt[T, :, :] = px1

    if idx_nodiff is not None:
        if zt is not None:
            xt = np.copy(zt)
        else:
            xt = np.zeros((T + 1, N, dim))
            xt_ = landmarkDirectEvolutionEuler(x0[idx_diff, :], at, Kpardiff, affine=affine,
                                               withPointSet=x0[idx_nodiff, :])
            xt[:, idx_diff, :] = np.copy(xt_[0])
            xt[:, idx_nodiff, :] = np.copy(xt_[1])
        for t in range(T, 0, -1):
            if withaff:
                pxt[t - 1, :, :] = np.dot(pxt[t,:,:], affineBasis.getExponential(timeStep * A[t - 1, :, :]))
            else:
                pxt[t-1, ...] = pxt[t, : ,:]
            pxt[t-1,idx_diff,:] += timeStep * (Kpardiff.applyDiffKT(xt[t-1,idx_diff,:], pxt[t,idx_diff,:],
                                                                   at[t-1,:,:], firstVar=xt[t-1,idx_diff,:],
                                                                   regWeight=regWeight, lddmm=True, dtype=dtype)
                                               + Kpardiff.applyDiffKT(xt[t-1,idx_nodiff,:], at[t-1,:,:],
                                                                   pxt[t,idx_nodiff,:], firstVar=xt[t-1,idx_diff,:],
                                                                   lddmm=False, dtype=dtype))
            pxt[t-1,idx_nodiff,:] += timeStep * Kpardiff.applyDiffKT(xt[t-1,idx_diff,:], pxt[t,idx_nodiff,:],
                                                                     at[t-1,:,:], firstVar=xt[t-1,idx_nodiff,:],
                                                                     lddmm=False, dtype=dtype)
    else:
        if zt is not None:
            xt = np.copy(zt)
        else:
            xt = landmarkDirectEvolutionEuler(x0, at, Kpardiff, affine=affine)
        for t in range(T, 0, -1):
            if withaff:
                pxt[t - 1, :, :] = np.dot(pxt[t,:,:], affineBasis.getExponential(timeStep * A[t - 1, :, :]))
            else:
                pxt[t - 1, ...] = pxt[t, : ,:]
            pxt[t-1,...] += timeStep * Kpardiff.applyDiffKT(xt[t-1,:,:], pxt[t,:,:], at[t-1,:,:], regWeight=regWeight,
                                                            lddmm=True, dtype=dtype)

    return pxt, xt


# Computes gradient after covariant evolution for deformation cost a^TK(x,x) a
def landmarkHamiltonianGradient(x0, at, px1, KparDiff, regWeight, getCovector=False, affine=None, zt=None, idx_diff=None, idx_nodiff=None):
    (pxt, xt) = landmarkHamiltonianCovector(x0, at, px1, KparDiff, regWeight, affine=affine, zt=zt, idx_diff=idx_diff, idx_nodiff=idx_nodiff)
    dat = np.zeros(at.shape)
    timeStep = 1.0/at.shape[0]
    if not (affine is None):
        A = affine[0]
        dA = np.zeros(affine[0].shape)
        db = np.zeros(affine[1].shape)

    for k in range(at.shape[0]):
        a = at[k, :, :]
        px = pxt[k+1, idx_diff, :]
        dat[k, :, :] = (2*regWeight*a-px)
        if not (affine is None):
            dA[k] = affineBasis.gradExponential(A[k] * timeStep, pxt[k + 1,:,:], xt[k,:,:])
            db[k] = pxt[k+1,:,:].sum(axis=0)

    if affine is None:
        if getCovector == False:
            return dat, xt
        else:
            return dat, xt, pxt
    else:
        if getCovector == False:
            return dat, dA, db, xt
        else:
            return dat, dA, db, xt, pxt


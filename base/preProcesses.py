import logging
import numpy as np
from numpy.linalg import solve
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors


def standardizeData(x, standardize='x', model='regression'):
    if standardize == 'x':
        mu0_x = np.mean(x, axis=0, keepdims=True)
        s0_x = np.std(x, axis=0, keepdims=True)
        s0_x[:, np.argwhere(s0_x == 0.0)[:, 1]] = 1.0
        x = np.divide(x - mu0_x, s0_x)
        return x, mu0_x, s0_x
    elif standardize == 'y':
        y = np.copy(x)
        if model == 'regression':
            mu0_y = np.mean(y, axis=0, keepdims=True)
            s0_y = np.std(y, axis=0, keepdims=True)
            s0_y[:, np.argwhere(s0_y == 0.0)[:, 1]] = 1.0
            y = np.divide(y - mu0_y, s0_y)
            return y, mu0_y, s0_y
        elif model == 'classification':  # calculate nClasses, wTr, and swTr
            nClasses = y.max() + 1
            nTr = np.array([(y == k).sum() for k in range(nClasses)])
            wTr = float(y.size) / (nTr[y[:, 0]] * nClasses)[:, np.newaxis]
            swTr = wTr.sum()
            return nClasses, wTr, swTr


def assignDimensions(layers, layerOutDim, dim_X, dim_Y, nD, nA, addDim_X, addDim_predictedY, ridgeType):
    logging.info('-' * 121)
    logging.info('addDim_X = {0:d}'.format(addDim_X))
    logging.info('addDim_predictedY = {0:d}'.format(addDim_predictedY))
    # assign dimensions to all layers
    numLayers = len(layers)
    if len(layerOutDim) == 1 and layerOutDim[0] is None:
        layerOutDim = layerOutDim * numLayers
    if len(layerOutDim) != numLayers:
        logging.info('\nlength of layerOutDim not equal to length of layers')
        exit()
    layer_A_indices = [i for i, val in enumerate(layers) if val == 'A']
    layer_D_indices = [i for i, val in enumerate(layers) if val == 'D']
    dim_A = [None] * nA
    dim_D = [None] * nD
    dim_all_layers = [None] * len(layers)
    ind_D = 0
    ind_A = 0
    # fix last layer(s) based on dim_Y and addDim_predictedY
    layerOutDim[-1] = dim_Y + addDim_predictedY
    for index in range(numLayers-1,-1,-1):
        if layers[index] == 'D':
            if layerOutDim[index-1] is None:
                layerOutDim[index-1] = layerOutDim[-1]
        else:
            break
    # go back to first layer and start assigning dimensions
    for index in range(numLayers):
        if index == 0:
            inputDim = dim_X + addDim_X
            if layerOutDim[index] is None:
                if layers[index+1] == 'D' and layerOutDim[index+1] is not None:
                    outputDim = layerOutDim[index+1]
                else:
                    outputDim = inputDim
                layerOutDim[index] = outputDim
            else:
                outputDim = layerOutDim[index]
        else:
            inputDim = layerOutDim[index-1]
            if layerOutDim[index] is None:
                if layers[index+1] == 'D' and layerOutDim[index+1] is not None:
                    outputDim = layerOutDim[index+1]
                else:
                    outputDim = inputDim
                layerOutDim[index] = outputDim
            else:
                outputDim = layerOutDim[index]
        if layers[index] == 'D':
            if inputDim == outputDim:
                dim_all_layers[index] = inputDim
                dim_D[ind_D] = inputDim
                ind_D += 1
            else:
                logging.info('\nlayer {0} is a D layer with unmatching input ({1}) and output ({2}) dimensions'.format(index,inputDim,outputDim))
                exit()
        elif layers[index] == 'A':
            dim_all_layers[index] = (inputDim + 1, outputDim)
            dim_A[ind_A] = (inputDim + 1, outputDim)
            ind_A += 1
    # check ridgeType for each A layer
    if len(ridgeType) != nA:
        logging.info('\nridgeType should have {0} entries'.format(nA))
        exit()
    if layers[-1] == 'A' and ridgeType[-1] != 'reg':
        logging.info('setting last A to ridgeType reg')
        ridgeType[-1] = 'reg'
    for layer_A in range(nA):
        if ridgeType[layer_A] != 'reg' and (dim_A[layer_A][0] - 1 != dim_A[layer_A][1]):
            logging.info('layer_A {0} not square; setting to ridgeType reg')
            ridgeType[layer_A] = 'reg'
    logging.info('updated layerOutDim = {0}'.format(layerOutDim))
    logging.info('updated ridgeType = {0}'.format(ridgeType))
    logging.info('dim_all_layers = {0}'.format(dim_all_layers))
    logging.info('-' * 121)
    return layerOutDim, dim_D, dim_A, dim_all_layers, layer_D_indices, layer_A_indices, ridgeType


def estimate_training_threshold(XTr, YTr, s0_y, prediction = 'regression'):
    d = XTr.shape[1]
    nn = min(2*d+1, XTr.shape[0]//5)
    nbrs = NearestNeighbors(n_neighbors=nn, metric='euclidean').fit(XTr)
    distx, I = nbrs.kneighbors(XTr)
    if prediction == 'regression':
        disty = np.zeros(YTr.shape[1])
        distg = 0
        # y = YTr[I[:, 1], :] - YTr[I[:, 0], :]
        # x = XTr[I[:, 1], :] - XTr[I[:, 0], :]
        # reg = LinearRegression(fit_intercept=False)
        # reg.fit(x,y)
        # yy = reg.predict(x)
        # rss = (((y-yy)**2).mean(axis=0) * s0_y[:, 0]**2).sum()
        #return rss/2

        grd = np.zeros((YTr.shape[0], YTr.shape[1], XTr.shape[1]))
        for i in range(XTr.shape[0]):
            X = XTr[I[i, 1:], :] - XTr[I[i, 0], :]
            Y = YTr[I[i, 1:], :] - YTr[I[i, 0], :]
            cov = 0.01*np.eye(d) + np.dot(X.T, X)
            covy = np.dot(X.T, Y)
            grd[i, :, :] = solve(cov, covy).T
            res = ((Y.T - np.dot(grd[i,:,:], X.T))**2).mean()  # Y --> Y.T
            distg += res

        distg /= XTr.shape[0]

        for k in range(YTr.shape[1]):
            # disty += ((YTr[I[:, 1], k] - YTr[I[:, 0], k])**2).mean() * s0_y[k, 0]**2
            # distg += (((grd[:, k, :] * (XTr[I[:, 1], :] - XTr[I[:, 0], :])).sum(axis=1))**2).mean() * s0_y[k, 0]**2
            u = ((YTr[I[:, 1], k] - YTr[I[:, 0], k] - (grd[:, k, :] * (XTr[I[:, 1], :] - XTr[I[:, 0], :])).sum(axis=1)) ** 2).mean()
            disty[k] +=  u * s0_y[0, k] ** 2   # s0_y[k, 0] ** 2  --> s0_y[0, k] ** 2

        return disty/2 #max(1e-6, (disty-distg)/2)
    elif prediction == 'classification':
        #return 0.01
        J = np.argsort(distx[:, 1])[:20]
        err1 = (YTr[I[J, 1], :] != YTr[I[J, 0], :]).mean()
        logging.info(f'error 1: {err1:.4f}')
        md = mode(YTr[I[:,1:6], 0], axis=1)
        J = md.count == md.count.max()
        pred = md.mode[J]
        #err = (YTr[I[J, 1], :] != YTr[I[J, 0], :]).mean()
        err = (YTr[J] != pred).mean()
        # err = (YTr[I[:, 1], 0] != YTr[I[:, 0], 0]).mean()
        return min(err, err1)



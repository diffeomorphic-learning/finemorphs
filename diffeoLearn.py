import os
from sys import platform
import logging
import copy
import time
import numpy as np
from sklearn.cluster import kmeans_plusplus
import matplotlib.pyplot as plt
import matplotlib
if 'DISPLAY' in os.environ or platform == 'darwin':
    matplotlib.use('qt5Agg')
    noDisplay = False
else:
    matplotlib.use("Agg")
    noDisplay = True
import pykeops
from base import loggingUtils
from base import pointSets as ps
from base import pointEvolution as evol, plotData
from base import conjugateGradient as cg, bfgs
from base import kernelFunctions as kfun, preProcesses as prep
from base.affineBasis import AffineBasis


class Direction:
    def __init__(self, vf_type='LDDMM', nD=1, nA=1):
        if vf_type == 'LDDMM':
            self.diff = [None] * nD
            self.aff = [None] * nD
            for layer_D in range(nD):
                self.diff[layer_D] = []
                self.aff[layer_D] = []
        self.upred = [None] * nA
        for layer_A in range(nA):
            self.upred[layer_A] = []


def __default_options__():
    opt = dict()

    opt['model'] = 'regression'         # 'regression' or 'classification'
    opt['vf_type'] = 'LDDMM'            # vector field type ('LDDMM')

    opt['trainSubset_flag'] = False     # True = only a subset of training data used in the diffeomorphism calculation
    opt['nSubset'] = 1000               # Number of data points in subset
    opt['KMeans_flag'] = True           # True = use KMeans to chose the subset of points

    opt['layers'] = ['A', 'D', 'A']     # order of A and D layers
    opt['layerOutDim'] = [None]         # number of dimensions to add to X before first layer
    opt['addDim_X'] = 1                 # number of dimensions to add to X before first layer
    opt['addDim_predictedY'] = 0        # = (dimension of outputted predicted Y) - (dimension of Y) (only for 'regression')

    opt['lam'] = 1.                     # regularization weight for each A layer
    opt['ridgeType'] = 'reg'            # ridge type for each A layer:  'reg', 'eye', 'sim'
    opt['gam'] = 1.                     # regularization weight for each D layer    (for vf_type == 'LDDMM')
    opt['Tsize'] = 10                   # number of time steps for each D layer

    opt['XTr_initFactor'] = 0.01        # XTr added dimension elements drawn from N(0,XTr_initFactor**2)
    opt['at_initFactor'] = 0.0          # initial elements of at ~ N(0,at_initFactor**2)    (for vf_type == 'LDDMM')
    opt['u_initFactor'] = 0.01          # initial elements of u[1:,:] ~ N(0,u_initFactor**2)   (if Xavier_flag = False)
    opt['Xavier_flag'] = True           # True = apply Xavier initialization to u:  u_initFactor = sqrt(1./d1))

    opt['affine'] = 'none'              # 'affine', 'similitude', 'euclidean', 'translation', 'diagonal', or 'none' (if 'LDDMM')
    opt['affineWeight'] = 10.           # multiplicative constant on affine regularization                          (if 'LDDMM')
    opt['rotWeight'] = 0.1              # multiplicative constant on rot regularization (supercedes affineWeight)   (if 'LDDMM')
    opt['scaleWeight'] = 10.            # multiplicative constant on scale regularization (supercedes affineWeight) (if 'LDDMM')
    opt['transWeight'] = 1.             # multiplicative constant on translation regularization (supercedes affineWeight) (if 'LDDMM')

    opt['name'] = 'lap'                 # kernel type ('gauss', 'lap', or 'min')
    opt['pykeops_flag'] = True          # True = use pykeops, False = use numba
    opt['dtype'] = 'float64'            # dtype for pykeops calculations
    opt['localMaps'] = None             #
    opt['affineK'] = None               # kernel affine component (None, 'affine', or 'euclidean')
    opt['w1'] = 1.0                     # weight for linear part (for affineK = 'affine' or 'euclidean')
    opt['w2'] = 1.0                     # weight for translation part (for affineK = 'affine' or 'euclidean')
    opt['center'] = None                # origin (for affineK = 'affine' or 'euclidean')
    opt['dim'] = 3                      # dimension (for affineK = 'euclidean')
    opt['order'] = 3                    # order for Laplacian kernel
    opt['kernelSigma'] = 0.5            # kernel widths for D layers

    opt['alg'] = 'bfgs'                 # optimization algorithm ('bfgs or 'cg')
    opt['maxIter'] = 2000               # max iterations in bfgs or conjugate gradient
    opt['burnIn'] = 20
    opt['affBurnIn'] = 100
    opt['coeffuPred'] = 1.              # value for all A layers
    opt['coeffAff'] = 1.                # value for all D layers    (for vf_type == 'LDDMM')
    opt['gradCoeff'] = 1.               # normalizing coefficient for gradient
    opt['gradEps'] = -1                 # stopping threshold for small gradient; if = -1, calculated in optimizeMatching
    opt['objEps'] = 1e-8                # stopping threshold for small variation in obj
    opt['epsInit'] = 0.1                # initial gradient step
    opt['memory'] = 25                  # for bfgs
    opt['Wolfe'] = True                 # for bfgs or cg
    opt['verb'] = True                  # for verbose printing

    opt['sigmaErrorFactor'] = 1.        # factor for sigmaError, the normalization for the error term in cost function
    opt['update_sigmaError'] = True     # True = update sigmaError iteratively based on training error

    opt['testGradient'] = False         # True = evaluates gradient accuracy over random direction (debug)

    opt['saveRate'] = 20                # rate for plotting, saving files, and calculating errors
    opt['logging_flag'] = False         # create log file
    opt['loggingFile'] = 'logging_info' # filename for log file
    opt['saveFile_flag'] = False        # save .vtk files and trajectory plots
    opt['saveFile'] = 'evolution'       # baseline filename for .vtk files
    opt['plot_flag'] = False            # True = plot results (Original, Truth, and Predict subplots) at saveRate
    opt['saveFigure_flag'] = False      # save figures
    opt['plot_ind'] = 0                 # data color:  index of y data for multivariate regression plots or .vtk files (automatically set to 0 for classif)
    opt['plot_layer_D'] = 0             # data positions:  index of D layer for x data in Truth and Predict plots or in .vtk files (if =-1, uses Original x data)
    opt['vmin_vmax_flag'] = False       # True = pin color vmin and vmax of all plots on subplot1 data (Original y dataset)

    return opt


class DiffeoLearn(object):

    ####################################################################################################################
    #   initialization start
    ####################################################################################################################

    def __init__(self, options=None):
        opt = __default_options__()
        if options is not None:
            for k in options.keys():
                opt[k] = options[k]
        self.__init_from_dict_(opt)


    def __init_from_dict_(self, opt):
        for k in opt.keys():
            setattr(self, k, opt[k])

        if noDisplay:
            self.plot_flag = False

        self.outputDir = os.getcwd() + '/output'
        if not os.path.exists(self.outputDir): os.makedirs(self.outputDir)

        if self.logging_flag:
            loggingUtils.setup_default_logging(self.outputDir, fileName=self.loggingFile, stdOutput=True, mode='w')  # 'w' = overwrite file (if exists)
        else:
            loggingUtils.setup_default_logging(stdOutput=False)

        self.numLayers = len(self.layers)
        self.nA = self.layers.count('A')  # number of A layers
        self.nD = self.layers.count('D')  # number of D layers
        if self.nA == 0 or self.nD == 0:
            logging.info(
                '\nmodel implementation requires at least one (non-identity) A layer and at least one (non-identity) D layer')
            exit()

        if not isinstance(self.lam, list): self.lam = [self.lam] * self.nA
        if not isinstance(self.ridgeType, list): self.ridgeType = [self.ridgeType] * self.nA
        if not isinstance(self.gam, list): self.gam = [self.gam] * self.nD
        if not isinstance(self.Tsize, list): self.Tsize = [self.Tsize] * self.nD

        if not isinstance(self.at_initFactor, list): self.at_initFactor = [self.at_initFactor] * self.nD
        if not isinstance(self.u_initFactor, list): self.u_initFactor = [self.u_initFactor] * self.nA
        if not isinstance(self.Xavier_flag, list): self.Xavier_flag = [self.Xavier_flag] * self.nA

        if not isinstance(self.affine, list): self.affine = [self.affine] * self.nD
        if not isinstance(self.affineWeight, list): self.affineWeight = [self.affineWeight] * self.nD
        if not isinstance(self.rotWeight, list): self.rotWeight = [self.rotWeight] * self.nD
        if not isinstance(self.scaleWeight, list): self.scaleWeight = [self.scaleWeight] * self.nD
        if not isinstance(self.transWeight, list): self.transWeight = [self.transWeight] * self.nD

        if not isinstance(self.name, list): self.name = [self.name] * self.nD
        if not isinstance(self.pykeops_flag, list): self.pykeops_flag = [self.pykeops_flag] * self.nD
        if not isinstance(self.dtype, list): self.dtype = [self.dtype] * self.nD
        if not isinstance(self.localMaps, list): self.localMaps = [self.localMaps] * self.nD
        if not isinstance(self.affineK, list): self.affineK = [self.affineK] * self.nD
        if not isinstance(self.w1, list): self.w1 = [self.w1] * self.nD
        if not isinstance(self.w2, list): self.w2 = [self.w2] * self.nD
        if not isinstance(self.center, list): self.center = [self.center] * self.nD
        if not isinstance(self.dim, list): self.dim = [self.dim] * self.nD
        if not isinstance(self.order, list): self.order = [self.order] * self.nD
        if not isinstance(self.kernelSigma, list): self.kernelSigma = [self.kernelSigma] * self.nD

        self.timeStep = [1. / item for item in self.Tsize]

        logging.info('-' * 121)
        logging.info('vf_type = {0}'.format(self.vf_type))

        # create kernels for each D layer ------------------------------------------------------------------------------
        if self.vf_type == 'LDDMM':
            if len(self.kernelSigma) != self.nD:
                logging.info('\nkernelSigma should have {0} entries'.format(self.nD))
                exit()
            logging.info('kernel sigmas = {}'.format(self.kernelSigma))
            self.KparDiff = [None] * self.nD
            for layer_D in range(self.nD):
                self.KparDiff[layer_D] = kfun.Kernel(name=self.name[layer_D], affine=self.affineK[layer_D],
                                                     sigma=self.kernelSigma[layer_D], order=self.order[layer_D],
                                                     w1=self.w1[layer_D], w2=self.w2[layer_D],
                                                     center=self.center[layer_D],
                                                     dim=self.dim[layer_D], localMaps=self.localMaps[layer_D],
                                                     dtype=self.dtype[layer_D], pykeops_flag=self.pykeops_flag[layer_D])
            if any(self.pykeops_flag):
                logging.info('pykeops.config.gpu_available = {0}'.format(pykeops.config.gpu_available))


        if self.plot_flag or self.saveFile_flag:
            if self.model == 'classification':
                self.plot_ind = 0
                logging.info('\nself.plot_ind set to 0')
            if self.plot_layer_D > self.nD - 1:
                logging.info('\nself.plot_layer_D must be < self.nD')
                exit()

        # close logging file
        logger = logging.getLogger()
        for h in logger.handlers:
            logger.removeHandler(h)


    def __init_plots__(self):
        if self.plot_flag or self.saveFile_flag:
            if self.plot_flag:
                self.figTrain = plt.figure('Training Data', figsize=(14, 6))
                self.figTrain.clf()
                mngr = plt.get_current_fig_manager()
                geom = mngr.window.geometry()
                x, y, dx, dy = geom.getRect()
                mngr.window.setGeometry(5, 30, dx, dy)
            if self.saveFile_flag:
                ps.savePoints(self.outputDir + '/Template_Train.vtk', self.XTr)
            self.plots_are_initialized = True


    def first_plot(self):
        if self.plot_flag or self.saveFile_flag:
            if not self.plots_are_initialized:
                self.__init_plots__()
            self.x3_orig_train = ps.build3DProjection(self.XTr[:, :], Y=self.YTr[:, :], mode=self.model)
            if self.plot_flag or self.saveFile_flag:
                self.colors_truth_train = np.copy(self.YTr[:, self.plot_ind])
            if self.saveFile_flag:
                ps.savePoints(self.outputDir + '/TrainingSet.vtk', self.x3_orig_train, scalars=self.colors_truth_train)
            if self.plot_flag:
                plotData.plotClassifRegress(self.x3_orig_train, self.colors_truth_train,
                                            self.x3_orig_train, self.colors_truth_train,
                                            self.x3_orig_train, self.colors_truth_train,
                                            fig=self.figTrain, title1='Original', vmin_vmax_flag=self.vmin_vmax_flag)
                plt.pause(0.1)


    def running_plot(self):
        if self.plot_flag or self.saveFile_flag:
            if not self.plots_are_initialized:
                self.__init_plots__()
            if self.saveFile_flag:
                if self.plot_layer_D == -1:
                    plotlayerD = 0
                else:
                    plotlayerD = np.copy(self.plot_layer_D)
                x3Tr_ = np.zeros((self.zt[plotlayerD].shape[0], self.zt[plotlayerD].shape[1], 3))
                for kk in range(self.Tsize[plotlayerD] + 1):
                    x3Tr_[kk,:,:] = ps.build3DProjection(self.zt[plotlayerD][kk, :, :], Y=self.YTr, mode=self.model)
                    ps.savePoints(self.outputDir + '/' + self.saveFile + str(kk) + '.vtk',
                                  x3Tr_[kk,:,:], scalars=self.colors_truth_train)
                fig_ = plt.figure()
                fig_.clf()
                ax_ = fig_.add_subplot(111, projection='3d')
                axes_ = [x3Tr_[:, :, 0].min(), x3Tr_[:, :, 0].max(), x3Tr_[:, :, 1].min(), x3Tr_[:, :, 1].max(),
                        x3Tr_[:, :, 2].min(), x3Tr_[:, :, 2].max()]
                for kk in range(self.Tsize[plotlayerD] + 1):
                    ax_.cla()
                    ax_.set_xlim3d(axes_[0], axes_[1])
                    ax_.set_ylim3d(axes_[2], axes_[3])
                    ax_.set_zlim3d(axes_[4], axes_[5])
                    cmap_ = plt.cm.jet
                    vmin_ = np.amin(np.ravel(self.colors_truth_train))
                    vmax_ = np.amax(np.ravel(self.colors_truth_train))
                    norm_ = matplotlib.colors.Normalize(vmin_, vmax_)
                    sm_ = plt.cm.ScalarMappable(cmap=cmap_, norm=norm_)
                    sm_.set_array([])
                    c_ = np.ravel(self.colors_truth_train)
                    ax_.scatter(x3Tr_[kk, :, 0], x3Tr_[kk, :, 1], x3Tr_[kk, :, 2], marker='o', s=2, c=cmap_(norm_(c_)))
                    plt.savefig(self.outputDir + '/train' + str(kk) + '.png')
                plt.close(fig_)
            if self.plot_flag:
                if self.plot_layer_D == -1:
                    x3Tr = np.copy(self.x3_orig_train)
                else:
                    x3Tr = ps.build3DProjection(self.zt[self.plot_layer_D][-1, :, :], Y=self.YTr, mode=self.model)
            if self.plot_flag:
                colors_predict = np.copy(self.guTr[:, self.plot_ind])
                plotData.plotClassifRegress(self.x3_orig_train, self.colors_truth_train,
                                            x3Tr, self.colors_truth_train, x3Tr, colors_predict,
                                            fig=self.figTrain, title1='Original', title2='Truth Y', title3='Predict Y',
                                            vmin_vmax_flag=self.vmin_vmax_flag)
                if self.saveFigure_flag:
                    plt.savefig(self.outputDir + '/train.png')
                time.sleep(0.5)

    ####################################################################################################################
    #   initialization stop
    ####################################################################################################################

    ####################################################################################################################
    #   diffeoLearn start
    ####################################################################################################################

    def fit(self, fit_x, fit_y, typeData='data', standardize_fit_x=True, standardize_fit_y=True, set_random_seed=False,
            pykeops_cleanup=False      # pykeops clean-up, in case old build files are still present
            ):

        if self.vf_type == 'LDDMM':
            if self.pykeops_flag and pykeops_cleanup:
                pykeops.clean_pykeops()

        if self.logging_flag:
            loggingUtils.setup_default_logging(self.outputDir, fileName=self.loggingFile, stdOutput=True, mode='a')  # 'a' = write without overwriting
        else:
            loggingUtils.setup_default_logging(stdOutput=False)

        logging.info('-' * 121)
        logging.info('\n{0:s}\n'.format(typeData))

        if set_random_seed:
            print('setting random seed')
            np.random.seed(42)

        # standardize x and y data -------------------------------------------------------------------------------------
        self.standardize_fit_x = standardize_fit_x
        self.standardize_fit_y = standardize_fit_y
        self.XTr = np.copy(fit_x)
        self.YTr = np.copy(fit_y)
        self.dim_X = self.XTr.shape[1]
        self.NTr = self.XTr.shape[0]
        if self.standardize_fit_x:
            self.XTr, self.mu0_x, self.s0_x = prep.standardizeData(self.XTr)
        if self.model == 'regression':
            self.dim_Y = self.YTr.shape[1]
            if self.standardize_fit_y:
                self.YTr, self.mu0_y, self.s0_y = prep.standardizeData(self.YTr, standardize='y')
        elif self.model == 'classification':
            self.nClasses, self.wTr, self.swTr = prep.standardizeData(self.YTr, standardize='y', model='classification')  # calculate nClasses, wTr, and swTr
            self.dim_Y = int(np.copy(self.nClasses))
            self.s0_y = None
            self.addDim_predictedY = 0
        logging.info('XTr.shape = {0}'.format(self.XTr.shape))
        logging.info('YTr.shape = {0}'.format(self.YTr.shape))


        # assign dimensions to D layers and A layers -------------------------------------------------------------------
        self.layerOutDim, self.dim_D, self.dim_A, self.dim_all_layers, self.layer_D_indices, self.layer_A_indices, self.ridgeType = \
            prep.assignDimensions(self.layers, self.layerOutDim, self.dim_X, self.dim_Y, self.nD, self.nA,
                                  self.addDim_X, self.addDim_predictedY, self.ridgeType)


        # modify u_initFactor if Xavier_flag == True -------------------------------------------------------------------
        for layer_A in range(self.nA):
            if self.Xavier_flag[layer_A]:
                d1 = self.dim_A[layer_A][0] - 1
                self.u_initFactor[layer_A] = np.sqrt(1. / d1)
        if any(self.Xavier_flag):
            logging.info('u_initFactor = {0}'.format([float("%.5f" % item) for item in self.u_initFactor]))
            logging.info('-' * 121)


        # estimate training threshold ----------------------------------------------------------------------------------
        _s = prep.estimate_training_threshold(self.XTr, self.YTr, self.s0_y, prediction=self.model)


        # initialize sigmaError ----------------------------------------------------------------------------------------
        # sigmaError: normalization for the error term
        if self.model == 'regression':
            self.sigmaError = ((self.NTr) ** .25) * max(np.sqrt((_s / self.s0_y ** 2).sum()), 0.1)
        elif self.model == 'classification':
            self.sigmaError = ((self.NTr) ** .5) * 0.01  # * max(_s, 0.01)  # max(np.sqrt(_s), 0.1)/data.s0_y.mean() #  NTr ** (1/4)
        self.sigmaError *= self.sigmaErrorFactor
        logging.info(f'sigmaError = {self.sigmaError:.6f}')


        # if only a subset of training data used in the diffeomorphism calculation, calculate the subset's indices:
        if self.trainSubset_flag:
            if self.KMeans_flag:
                centers, self.idx_diff = kmeans_plusplus(self.XTr, n_clusters=self.nSubset)
                self.idx_nodiff = np.setdiff1d(np.arange(self.NTr), self.idx_diff)
            else:
                idx = np.random.permutation(self.NTr)
                self.idx_diff = idx[:self.nSubset]
                self.idx_nodiff = idx[self.nSubset:]
            logging.info('train subset size = {0}'.format(self.nSubset))
        else:
            self.nSubset = self.NTr
            self.idx_diff = np.arange(self.NTr)
            self.idx_nodiff = None


        # modify self.XTr (i.e., add dimension(s)), if applicable ------------------------------------------------------
        # if addDim_X>0, add dimensions to self.XTr and self.XTe
        if self.addDim_X > 0:
            addCol = np.zeros((self.NTr, self.addDim_X))
            addCol[self.idx_diff, :] = self.XTr_initFactor * np.random.normal(size=(self.nSubset, self.addDim_X))
            self.XTr = np.concatenate((self.XTr, addCol), axis=1)
            logging.info('updated XTr.shape = ({0:d},{1:d})'.format(self.XTr.shape[0], self.XTr.shape[1]))


        # initialize plots, algorithm, and variables -------------------------------------------------------------------
        if self.plot_flag or self.saveFile_flag:
            if self.plot_ind > self.dim_Y:
                logging.info('\nself.plot_ind must be <= self.dim_Y')
                exit()

        self.plots_are_initialized = False

        self.resetAlgorithm(self.alg)

        self.iter = 0

        if self.vf_type == 'LDDMM':
            self.affB = [None] * self.nD
            self.affineDim = [0] * self.nD
            self.affineBasis = [None] * self.nD
            for layer_D in range(self.nD):
                self.affB[layer_D] = AffineBasis(self.dim_D[layer_D], self.affine[layer_D])
                self.affineDim[layer_D] = self.affB[layer_D].affineDim
                self.affineBasis[layer_D] = self.affB[layer_D].basis
                aw = 1.  # self.dim_D[layer_D] #np.sqrt(self.dim_D[layer_D])
                self.affineWeight[layer_D] = self.affineWeight[layer_D] * np.ones([self.affineDim[layer_D], 1]) / aw
                if (len(self.affB[layer_D].rotComp) > 0) & (self.rotWeight[layer_D] != None):
                    self.affineWeight[layer_D][self.affB[layer_D].rotComp] = self.rotWeight[layer_D] / aw
                if (len(self.affB[layer_D].simComp) > 0) & (self.scaleWeight[layer_D] != None):
                    self.affineWeight[layer_D][self.affB[layer_D].simComp] = self.scaleWeight[layer_D] / aw
                if (len(self.affB[layer_D].transComp) > 0):
                    if (self.transWeight[layer_D] != None):
                        self.affineWeight[layer_D][self.affB[layer_D].transComp] = self.transWeight[layer_D]
                    else:
                        self.affineWeight[layer_D][self.affB[layer_D].transComp] *= aw

        if self.vf_type == 'LDDMM':
            self.at = [None] * self.nD
            self.Afft = [None] * self.nD
            for layer_D in range(self.nD):
                self.at[layer_D] = self.at_initFactor[layer_D] * np.random.normal(
                    size=(self.Tsize[layer_D], self.nSubset, self.dim_D[layer_D]))
                self.Afft[layer_D] = np.zeros((self.Tsize[layer_D], self.affineDim[layer_D]))
            self.Kdtype = []
            for K in self.KparDiff:
                self.Kdtype.append(K.dtype)
            self.v = [None] * self.nD
            for layer_D in range(self.nD):
                self.v[layer_D] = np.zeros((self.Tsize[layer_D] + 1, self.nSubset, self.dim_D[layer_D]))

        self.u = [None] * self.nA
        self.ridgecost = [None] * self.nA
        self.ridgecostgradient = [None] * self.nA
        for layer_A in range(self.nA):
            if self.ridgeType[layer_A] == 'reg':
                self.ridgecost[layer_A] = ps._RidgeSumOfSquares
                self.ridgecostgradient[layer_A] = ps._RidgeSumOfSquaresGradient
            elif self.ridgeType[layer_A] == 'eye':
                self.ridgecost[layer_A] = ps._RidgeSumOfSquaresToEye
                self.ridgecostgradient[layer_A] = ps._RidgeSumOfSquaresToEyeGradient
            elif self.ridgeType[layer_A] == 'sim':
                self.ridgecost[layer_A] = ps._RidgeSumOfSquaresToSim
                self.ridgecostgradient[layer_A] = ps._RidgeSumOfSquaresToSimGradient
            if self.ridgeType[layer_A] == 'reg':
                self.u[layer_A] = np.concatenate(
                    (np.zeros((1, self.dim_A[layer_A][1])),
                     self.u_initFactor[layer_A] * np.random.normal(size=(self.dim_A[layer_A][0] - 1,
                                                                         self.dim_A[layer_A][1]))), axis=0)
            elif self.ridgeType[layer_A] in ('eye','sim'):
                self.u[layer_A] = np.concatenate(
                    (np.zeros((1, self.dim_A[layer_A][1])),
                     np.eye(self.dim_A[layer_A][0] - 1) +
                     np.multiply(np.eye(self.dim_A[layer_A][0] - 1),
                                 self.u_initFactor[layer_A] * np.random.normal(size=(self.dim_A[layer_A][0] - 1,
                                                                                     self.dim_A[layer_A][1])))), axis=0)
        if self.model == 'classification':
            self.u[-1][:, 0] = 0

        self.obj = None
        self.u_sVals = [None] * self.nA  # singular values of u matrices
        self.obj_array = np.array([])
        self.u_sVals_array = [np.array([])] * self.nA
        self.trainError_array = np.array([])


        # initialize trajectories --------------------------------------------------------------------------------------
        if self.vf_type == 'LDDMM':
            obj, self.zt, self.forwardState_In, self.forwardState_Out = self.objectiveFunDef_LDDMM(self.at, self.Afft, withTrajectory=True)

        if self.model == 'regression':
            self.guTr = np.copy(self.forwardState_Out[-1][:, :self.dim_Y])
        elif self.model == 'classification':
            self.guTr = np.argmax(self.forwardState_Out[-1], axis=1)[:, np.newaxis]
        logging.info('-' * 121)
        if not self.plots_are_initialized:
            self.first_plot()


        # start fit ----------------------------------------------------------------------------------------------------
        # update_sigmaError
        if self.update_sigmaError:
            if self.model == 'regression':
                target_train_err = max(_s.sum(), 0.01)
            elif self.model == 'classification':
                target_train_err = max(min(_s, 0.1), 0.01)
            logging.info(f'Target training error: {target_train_err:0.6f}')
            objEps_ = np.copy(self.objEps)
            maxIter_ = np.copy(self.maxIter)
            self.objEps = 1e-4
            self.maxIter = 2000
            self.optimizeMatching()
            nloop = 10
            loop = 1
            while loop < nloop and self.train_err > target_train_err:
                self.sigmaError /= 1.4
                logging.info(f'sigmaError = {self.sigmaError:.6f}')
                self.resetAlgorithm(self.alg)
                self.optimizeMatching()
                loop += 1
            self.maxIter = maxIter_
            self.objEps = objEps_
            self.resetAlgorithm(self.alg)
        self.optimizeMatching()

        # close logging file
        logger = logging.getLogger()
        for h in logger.handlers:
            logger.removeHandler(h)


    def predict(self, predict_x, predict_y=None):
        if self.logging_flag:
            loggingUtils.setup_default_logging(self.outputDir, fileName=self.loggingFile, stdOutput=True, mode='a')  # 'a' = write without overwriting
        else:
            loggingUtils.setup_default_logging(stdOutput=False)

        self.XPr = np.copy(predict_x)
        self.NPr = predict_x.shape[0]
        self.forwardStatePr_In = [None] * self.numLayers
        self.forwardStatePr_Out = [None] * self.numLayers
        self.ztPr = [None] * self.nD
        if self.standardize_fit_x:
            self.XPr = np.divide(self.XPr - self.mu0_x, self.s0_x)
        if self.addDim_X > 0:
            self.XPr = np.concatenate((self.XPr, np.zeros((self.NPr, self.addDim_X))), axis=1)

        if self.vf_type == 'LDDMM':
            A = self.getAffine()
        zPr = np.copy(self.XPr)
        for layer in range(self.numLayers):
            if self.layers[layer] == 'D':
                self.forwardStatePr_In[layer] = np.copy(zPr)
                layer_D = self.layer_D_indices.index(layer)
                if self.vf_type == 'LDDMM':
                    testRes = evol.landmarkDirectEvolutionEuler(self.zt[layer_D][0, self.idx_diff, :],
                                                                self.at[layer_D],
                                                                self.KparDiff[layer_D],
                                                                affine=A[layer_D],
                                                                withPointSet=zPr,
                                                                onlyPointSet=True,
                                                                zt=np.copy(self.zt[layer_D][:, self.idx_diff, :]))
                                                                # if onlyPointSet=True, testRes[0] is not propagated
                                                                # but instead set to optional parameter zt
                    self.ztPr[layer_D] = np.copy(testRes[1])
                zPr = np.copy(self.ztPr[layer_D][-1, :, :])
                self.forwardStatePr_Out[layer] = np.copy(zPr)
            else:
                self.forwardStatePr_In[layer] = np.copy(zPr)
                layer_A = self.layer_A_indices.index(layer)
                zPr = np.concatenate((np.ones((zPr.shape[0], 1)), zPr), axis=1)
                zPr = np.dot(zPr, self.u[layer_A])
                self.forwardStatePr_Out[layer] = np.copy(zPr)
        if self.model == 'regression':
            self.guPr = np.copy(self.forwardStatePr_Out[-1][:, :self.dim_Y])
        elif self.model == 'classification':
            self.guPr = np.argmax(self.forwardStatePr_Out[-1], axis=1)[:, np.newaxis]

        self.predict_results = np.copy(self.guPr)
        if (self.model == 'regression') and (self.standardize_fit_y):
            self.predict_results = np.multiply(self.predict_results, self.s0_y) + self.mu0_y

        if predict_y is not None:
            logging.info('-' * 121)
            if self.model == 'regression':
                self.predict_err = ((self.predict_results - predict_y) ** 2).sum() / predict_y.shape[0]
            elif self.model == 'classification':
                nPr = np.array([(predict_y == k).sum() for k in range(self.nClasses)])
                wPr = float(predict_y.size) / (nPr[predict_y[:, 0]] * self.nClasses)[:, np.newaxis]
                swPr = wPr.sum()
                self.predict_err = np.sum(np.not_equal(self.predict_results, predict_y) * wPr) / swPr
            logging.info('Prediction Error {0:.8f}'.format(self.predict_err))

        # close logging file
        logger = logging.getLogger()
        for h in logger.handlers:
            logger.removeHandler(h)

        return self.predict_results

    ####################################################################################################################
    #   diffeoLearn stop
    ####################################################################################################################

    ####################################################################################################################
    #   optimizeMatching start
    ####################################################################################################################

    def optimizeMatching(self):
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        if self.alg == 'bfgs':
            [grd2] = self.dotProduct_euclidean(grd, [grd])
        else:
            [grd2] = self.dotProduct(grd, [grd])
        if self.gradEps < 0:
            self.gradEps = max(0.00001, np.sqrt(grd2) / 10000000)
            # self.gradEps = max(0.0001, np.sqrt(grd2) / 10000)
        self.coeffAff = self.coeffAff1
        if self.alg == 'bfgs':
            bfgs.bfgs(self, verb=self.verb, maxIter=self.maxIter, TestGradient=self.testGradient,
                      epsInit=self.epsInit, memory=self.memory, Wolfe=self.Wolfe)
        else:
            cg.cg(self, verb=self.verb, maxIter=self.maxIter, TestGradient=self.testGradient,
                  epsInit=self.epsInit)


    def resetAlgorithm(self, algorithm):
        if algorithm == 'cg':
            self.alg = 'cg'
            for layer_D in range(self.nD):
                if self.KparDiff[layer_D].localMaps in (None, 'predict'):
                    self.coeffuPred = 1.    # value for all A layers
                    self.coeffAff1 = 1.
                    self.coeffAff2 = 1.
                else:
                    self.coeffuPred = 1.    # value for all A layers
                    self.coeffAff1 = 1.
                    self.coeffAff2 = 1.
            self.coeffAff = self.coeffAff1  # value for all D layers (for vf_type == 'LDDMM')
            if self.Wolfe:
                self.euclideanGradient = True
            else:
                self.euclideanGradient = False
        else:
            self.alg = 'bfgs'
            self.coeffuPred = 1.            # value for all A layers
            self.coeffAff1 = 1.
            self.coeffAff2 = 1.
            self.coeffAff = self.coeffAff1  # value for all D layers (for vf_type == 'LDDMM')
            self.euclideanGradient = True
        if self.trainSubset_flag:
            self.euclideanGradient = True
        self.reset = True

    ####################################################################################################################
    #   optimizeMatching stop
    ####################################################################################################################

    ####################################################################################################################
    #   gradientTools start
    ####################################################################################################################

    def testGradientFun(self, obj, grd, gradCoeff, opt=None, dotProduct=None):
        dirfoo = opt.randomDir()
        epsfoo = 1e-4  # 1e-6 (in bfgs)
        objfoo1 = opt.updateTry(dirfoo, epsfoo, obj - 1e10)
        [grdfoo] = dotProduct(grd, [dirfoo])
        ## logging.info('Test Gradient: %.6f %.6f' %((objfoo - obj)/epsfoo, -grdfoo * gradCoeff))
        objfoo1_ = opt.updateTry(dirfoo, -epsfoo, obj - 1e10)
        logging.info('Test Gradient: %.6f %.6f' % ((objfoo1 - objfoo1_) / (2.0 * epsfoo), -grdfoo * gradCoeff))
        objfoo2 = opt.updateTry(dirfoo, 2.0 * epsfoo, obj - 1e10)
        objfoo2_ = opt.updateTry(dirfoo, -2.0 * epsfoo, obj - 1e10)
        logging.info('Test Gradient: %.6f %.6f' % (
            (-objfoo2 + 8.0 * objfoo1 - 8.0 * objfoo1_ + objfoo2_) / (12.0 * epsfoo), -grdfoo * gradCoeff))
        objfoo3 = opt.updateTry(dirfoo, 3.0 * epsfoo, obj - 1e10)
        objfoo3_ = opt.updateTry(dirfoo, -3.0 * epsfoo, obj - 1e10)
        logging.info('Test Gradient: %.6f %.6f' % (
            (objfoo3 - 9.0 * objfoo2 + 45.0 * objfoo1 - 45.0 * objfoo1_ + 9.0 * objfoo2_ - objfoo3_) / (60.0 * epsfoo),
            -grdfoo * gradCoeff))
        objfoo4 = opt.updateTry(dirfoo, 4.0 * epsfoo, obj - 1e10)
        objfoo4_ = opt.updateTry(dirfoo, -4.0 * epsfoo, obj - 1e10)
        logging.info('Test Gradient: %.6f %.6f' % ((
                                                           -objfoo4 + 32.0 / 3.0 * objfoo3 - 56.0 * objfoo2 + 224.0 * objfoo1 - 224.0 * objfoo1_ + 56.0 * objfoo2_ - 32.0 / 3.0 * objfoo3_ + objfoo4_) /
                                                   (280.0 * epsfoo), -grdfoo * gradCoeff))


    def getGradient(self, coeff=1.0, update=None):
                                                # called by optimizeMatching, cg, and bfgs
                                                # returns grd.upred, grd.diff, and grd.aff
                                                # getGradient(coeff) returns coeff * gradient; the result can be used as 'direction' in updateTry
                                                # (if update is not None, updated form = [u, zt, forwardState_In, forwardState_Out, at, Afft, dir, eps]
                                                # (note:  zt, forwardState_In, and forwardState_Out are NOT variables;
                                                #         however, for bfgs, providing them in the 'updated' parameters
                                                #         allows us to bypass unnecessary propagation)
        if update is None:       # True for cg and parts of bfgs
            u = copy.deepcopy(self.u)
            zt = copy.deepcopy(self.zt)
            forwardState_In = copy.deepcopy(self.forwardState_In)
            forwardState_Out = copy.deepcopy(self.forwardState_Out)
            if self.vf_type == 'LDDMM':
                at = copy.deepcopy(self.at)
                Afft = copy.deepcopy(self.Afft)
                A = self.getAffine(Afft=Afft)
        else:                   # only in bfgs for saving intermediate tries during line search
            if (update[0] is self.updated_lineSearch[6]) and (update[1] == self.updated_lineSearch[7]):
                updated = self.updated_lineSearch
                u = copy.deepcopy(updated[0])
                zt = copy.deepcopy(updated[1])
                forwardState_In = copy.deepcopy(updated[2])
                forwardState_Out = copy.deepcopy(updated[3])
                if self.vf_type == 'LDDMM':
                    at = copy.deepcopy(updated[4])
                    Afft = copy.deepcopy(updated[5])
                    A = self.getAffine(Afft=Afft)
            else:
                dir = update[0]
                eps = update[1]
                uTry = [(self.u[layer_A] - eps * dir.upred[layer_A]) for layer_A in range(self.nA)]
                if self.vf_type == 'LDDMM':
                    atTry = [(self.at[layer_D] - eps * dir.diff[layer_D]) for layer_D in range(self.nD)]
                    AfftTry = copy.deepcopy(self.Afft)
                    AfftTry = [
                        (self.Afft[layer_D] - eps * dir.aff[layer_D]) if self.affineDim[layer_D] > 0 else AfftTry[
                            layer_D] for layer_D in range(self.nD)]
                    foo = self.objectiveFunDef_LDDMM(atTry, AfftTry, u=uTry,
                                                     withTrajectory=True)
                    at = copy.deepcopy(atTry)
                    Afft = copy.deepcopy(AfftTry)
                    A = self.getAffine(Afft=Afft)
                ztTry = copy.deepcopy(foo[1])
                forwardStateInTry = copy.deepcopy(foo[2])
                forwardStateOutTry = copy.deepcopy(foo[3])
                u = copy.deepcopy(uTry)
                zt = copy.deepcopy(ztTry)
                forwardState_In = copy.deepcopy(forwardStateInTry)
                forwardState_Out = copy.deepcopy(forwardStateOutTry)

        grd = Direction(vf_type=self.vf_type, nD=self.nD, nA=self.nA)

        # calculate grd.upred w.r.t. ridge regularization term
        grd.upred = ps.RidgeGradientInU(u, lam=self.lam, ridgecostgradient=self.ridgecostgradient)

        # start backpropagation and calculation of grd.diff, grd.aff, and the rest of grd.upred
        finalState = forwardState_Out[-1]
        backpropState_In = [None] * self.numLayers
        backpropState_Out = [None] * self.numLayers
        if self.model == 'regression':
            backpropState_In[-1] = -ps.RegressionScoreGradient(finalState, self.YTr, sigmaError=self.sigmaError)
        elif self.model == 'classification':
            backpropState_In[-1] = -ps.LogisticScoreGradient(finalState, self.YTr, w=self.wTr, sigmaError=self.sigmaError)
        foo = [None] * self.nD
        for layer in range(self.numLayers-1,-1,-1):
            if self.layers[layer] == 'A':
                layer_A = self.layer_A_indices.index(layer)

                # calculate backpropState_Out of current layer and backpropState_In of (layer-1)
                backpropState_Out[layer] = np.dot(backpropState_In[layer], u[layer_A][1:, :].T)
                if layer > 0:
                    backpropState_In[layer-1] = np.copy(backpropState_Out[layer])

                # calculate grd.upred of current layer
                forwardStateIn_with_ones = np.concatenate((np.ones((forwardState_In[layer].shape[0], 1)), forwardState_In[layer]), axis=1)
                grd.upred[layer_A] += -np.dot(forwardStateIn_with_ones.T, backpropState_In[layer])
                grd.upred[layer_A] = grd.upred[layer_A] / (coeff * self.coeffuPred)

            else:
                layer_D = self.layer_D_indices.index(layer)

                # calculate backpropState_Out of current layer, backpropState_In of (layer-1), grd.diff, and grd.aff
                if self.vf_type == 'LDDMM':
                    foo[layer_D] = self.hamiltonianGradient_LDDMM(backpropState_In[layer], at[layer_D], A[layer_D],
                                                                  self.KparDiff[layer_D],
                                                                  self.gam[layer_D],
                                                                  x0=zt[layer_D][0, :, :],
                                                                  zt=zt[layer_D][:, :, :],
                                                                  idx_diff=self.idx_diff,
                                                                  idx_nodiff=self.idx_nodiff)
                                            # hamiltonianGradient_LDDMM(...) = dat, xt, pxt (if affine=None and getCovector=True)
                                            # = dat, dA, db, xt, pxt (if affine not None and getCovector=True)
                                            # includes pt (by propagating p(1) backwards to p(0)) and dat (= p-2a)
                    if self.euclideanGradient:  # True for bfgs or trainSubset_flag=True
                        grd.diff[layer_D] = np.zeros(foo[layer_D][0].shape)
                        for t in range(self.Tsize[layer_D]):
                            z_ = zt[layer_D][t, self.idx_diff, :]
                            grd.diff[layer_D][t, :, :] = \
                                self.KparDiff[layer_D].applyK(z_,foo[layer_D][0][t, :, :]) / (coeff * self.Tsize[layer_D])
                            if self.trainSubset_flag:
                                z__ = zt[layer_D][t, self.idx_nodiff, :]
                                grd.diff[layer_D][t, :, :] -= \
                                    self.KparDiff[layer_D].applyK(z__,foo[layer_D][-1][t+1, self.idx_nodiff, :],firstVar=z_) / (coeff * self.Tsize[layer_D])
                    else:  # True for cg with Wolfe=True and trainSubset_flag=False
                        grd.diff[layer_D] = foo[layer_D][0] / (coeff * self.Tsize[layer_D])
                    dim2 = self.dim_D[layer_D] ** 2
                    grd.aff[layer_D] = np.zeros(Afft[layer_D].shape)
                    if self.affineDim[layer_D] > 0:
                        dA = foo[layer_D][1]
                        db = foo[layer_D][2]
                        grd.aff[layer_D] = 2 * self.affineWeight[layer_D].reshape([1, self.affineDim[layer_D]]) * Afft[layer_D]
                        for t in range(self.Tsize[layer_D]):
                            dAff = self.affineBasis[layer_D].T.dot(np.vstack(
                                [dA[t].reshape([dim2, 1]), db[t].reshape([self.dim_D[layer_D], 1])]))
                            grd.aff[layer_D][t] -= dAff.reshape(grd.aff[layer_D][t].shape)
                        grd.aff[layer_D] /= (self.coeffAff * coeff * self.Tsize[layer_D])
                backpropState_Out[layer] = np.copy(foo[layer_D][-1][0, :, :])  # foo[layer_D][-1].shape = < Tsize[layer_D], NTr, dim_D[layer_D] >
                if layer > 0:
                    backpropState_In[layer-1] = np.copy(backpropState_Out[layer])

        return grd


    def hamiltonianGradient_LDDMM(self, px1, at, affine, kernel, regWeight, x0=None, zt=None, idx_diff=None, idx_nodiff=None):
                                # called by getGradient:
                                # return dat, xt, pxt (if affine=None and getCovector=True)
                                # return dat, dA, db, xt, pxt (if affine not None and getCovector=True)
                                # i.e., includes pxt (by propagating p(1) backwards to p(0)) and dat (= p-2a)
        if x0 is None:
            if zt is not None:
                x0 = zt[0,:,:]
            else:
                print('\nerror in hamiltonianGradient_LDDMM')
                exit()
        return evol.landmarkHamiltonianGradient(x0, at, px1, kernel, regWeight, getCovector=True,
                                                affine=affine, zt=zt, idx_diff=idx_diff, idx_nodiff=idx_nodiff)

    ####################################################################################################################
    #   gradientTools stop
    ####################################################################################################################

    ####################################################################################################################
    #   objectiveFunction start
    ####################################################################################################################

    def objectiveFun(self):  # called by cg and bfgs
                             # if self.obj == None, updates self.obj, self.zt, self.forwardState_In, and
                             #                self.forwardState_Out, and returns updated self.obj
                             # otherwise, returns current self.obj
        if self.obj == None:
            if self.vf_type == 'LDDMM':
                (self.obj, self.zt, self.forwardState_In, self.forwardState_Out) \
                    = self.objectiveFunDef_LDDMM(self.at, self.Afft, withTrajectory=True)
            self.obj += self.dataTerm(self.forwardState_Out[-1])
            self.obj += self.uTerm(self.u)
        return self.obj


    def dataTerm(self, _finalState):    # called by objectiveFun and updateTry
                                        # calculates RegressionScore or LogisticScore
        if self.model == 'regression':
            obj = ps.RegressionScore(_finalState, self.YTr, sigmaError=self.sigmaError)
        elif self.model == 'classification':
            obj = ps.LogisticScore(_finalState, self.YTr, w=self.wTr, sigmaError=self.sigmaError)
        return obj


    def uTerm(self, _u):        # called by objectiveFun and updateTry
                                # calculates ridge regularization term
        if self.model == 'regression':
            obj = ps.RidgeRegularization(_u, lam=self.lam, ridgecost=self.ridgecost)
        elif self.model == 'classification':
            obj = ps.RidgeRegularization(_u, lam=self.lam, ridgecost=self.ridgecost)
        return obj


    def objectiveFunDef_LDDMM(self, at, Afft, kernel=None, withTrajectory=False, withJacobian=False,
                              x0=None, u=None, gam=None):
                            # called by objectiveFun, updateTry, fit, and endOfIteration
                            # calculates and returns obj_LDDMM,
                            #                        zt, forwardStateIn, forwardStateOut (if withTrajectory=True),
                            #                    and Jt (if withJacobian=True)
        if gam is None:
            gam = self.gam
        if x0 is None:
            x0 = self.XTr
        if kernel is None:
            kernel = self.KparDiff
        if u is None:
            u = self.u
        zt = [None] * self.nD
        for layer_D in range(self.nD):
            zt[layer_D] = np.zeros((self.Tsize[layer_D] + 1, self.NTr, self.dim_D[layer_D]))
        Jt = [None] * self.nD
        forwardStateIn = [None] * self.numLayers
        forwardStateOut = [None] * self.numLayers
        obj = 0
        A = self.getAffine(Afft=Afft)
        zTr = np.copy(x0)
        for layer in range(self.numLayers):
            forwardStateIn[layer] = np.copy(zTr)
            if self.layers[layer] == 'D':
                layer_D = self.layer_D_indices.index(layer)
                if self.trainSubset_flag:
                    trainRes = evol.landmarkDirectEvolutionEuler(zTr[self.idx_diff, :], at[layer_D], kernel[layer_D],
                                                                 affine=A[layer_D],
                                                                 withJacobian=withJacobian, withNormals=None,
                                                                 withPointSet=zTr[self.idx_nodiff, :])
                    zt[layer_D][:, self.idx_diff, :] = np.copy(trainRes[0])
                    zt[layer_D][:, self.idx_nodiff, :] = np.copy(trainRes[1])
                    if withJacobian:
                        Jt[layer_D] = np.copy(trainRes[2])
                else:
                    trainRes = evol.landmarkDirectEvolutionEuler(zTr, at[layer_D], kernel[layer_D], affine=A[layer_D],
                                                                 withJacobian=withJacobian, withNormals=None)
                    if withJacobian:
                        zt[layer_D] = np.copy(trainRes[0])
                        Jt[layer_D] = np.copy(trainRes[1])
                    else:
                        zt[layer_D] = np.copy(trainRes)
                zTr = np.copy(zt[layer_D][-1,:,:])
                obj1 = 0
                for t in range(self.Tsize[layer_D]):
                    z_ = zt[layer_D][t, self.idx_diff, :] # z_ = zt[layer_D][t, :, :]
                    a = at[layer_D][t, :, :]
                    ra = kernel[layer_D].applyK(z_, a)
                    if hasattr(self, 'v'):
                        self.v[layer_D][t, :] = ra
                    obj = obj + gam[layer_D] * self.timeStep[layer_D] * np.multiply(a, (ra)).sum()
                    if self.affineDim[layer_D] > 0:
                        obj1 += self.timeStep[layer_D] \
                                * np.multiply(self.affineWeight[layer_D].reshape(Afft[layer_D][t].shape), Afft[layer_D][t] ** 2).sum()
                obj += obj1
            else:
                layer_A = self.layer_A_indices.index(layer)
                zTr = np.concatenate((np.ones((zTr.shape[0], 1)), zTr), axis=1)
                zTr = np.dot(zTr, u[layer_A])
            forwardStateOut[layer] = np.copy(zTr)

        if withJacobian:
            return obj, zt, forwardStateIn, forwardStateOut, Jt
        elif withTrajectory:
            return obj, zt, forwardStateIn, forwardStateOut
        else:
            return obj


    def getAffine(self, Afft=None):
        A = [None] * self.nD
        if Afft is None:
            Afft_ = copy.deepcopy(self.Afft)
        else:
            Afft_ = copy.deepcopy(Afft)
        A = [self.affB[layer_D].getTransforms(Afft_[layer_D]) if self.affineDim[layer_D] > 0 else A[layer_D] for layer_D in range(self.nD)]
        return A

    ####################################################################################################################
    #   objectiveFunction stop
    ####################################################################################################################

    ####################################################################################################################
    #   optimizationTools start
    ####################################################################################################################

    def getVariable(self):
        if self.vf_type == 'LDDMM':
            return [self.at, self.Afft, self.u]


    def acceptVarTry(self):
        if self.vf_type == 'LDDMM':
            self.at = copy.deepcopy(self.atTry)
            self.Afft = copy.deepcopy(self.AfftTry)
        self.obj = np.copy(self.objTry)
        self.u = copy.deepcopy(self.uTry)
        self.zt = copy.deepcopy(self.ztTry)                             # zt, forwardState_In, and forwardState_Out
        self.forwardState_In = copy.deepcopy(self.forwardStateInTry)    #   are NOT variables; however, updating them
        self.forwardState_Out = copy.deepcopy(self.forwardStateOutTry)  #   here bypasses unnecessary propagation later


    def updateTry(self, dir, eps, objRef=None):
                            # called by cg and bfgs
                            # takes a step in direction dir (grd) with step size eps (learning rate)
                            # and calculates atTry, AfftTry, objTry, and uTry  (for self.vf_type == 'LDDMM')
                            # returns objTry
                            # saves [uTry, ztTry, forwardStateInTry, forwardStateOutTry, atTry, AfftTry, dir, eps]
                            #   in self.updated_lineSearch for use in bfgs linesearch
                            # if objTry < objRef or objRef == None, updates
                            #        self.atTry, self.AfftTry, self.objTry, self.uTry, self.ztTry, self.forwardStateInTry, self.forwardStateOutTry
                            # (note:  ztTry, forwardStateInTry, and forwardStateOutTry are NOT variables;
                            #         however, updating them here bypasses unnecessary propagation later)
        objTry = 0
        uTry = [(self.u[layer_A] - eps * dir.upred[layer_A]) for layer_A in range(self.nA)]
        if self.vf_type == 'LDDMM':
            atTry = [(self.at[layer_D] - eps * dir.diff[layer_D]) for layer_D in range(self.nD)]
            AfftTry = copy.deepcopy(self.Afft)
            AfftTry = [(self.Afft[layer_D] - eps * dir.aff[layer_D]) if self.affineDim[layer_D] > 0 else AfftTry[layer_D] for layer_D in range(self.nD)]
            foo = self.objectiveFunDef_LDDMM(atTry, AfftTry, u=uTry, withTrajectory=True)   # propagate each zt[layer_D] trajectory forward,
                                                                                            # then return obj_LDDMM, trajectories, and forward states
        objTry += foo[0]  # add obj_LDDMM to objTry
        forwardStateInTry = copy.deepcopy(foo[2])
        forwardStateOutTry = copy.deepcopy(foo[3])
        finalStateTry = forwardStateOutTry[-1]
        dataTermTry = self.dataTerm(finalStateTry)
        uTermTry = self.uTerm(uTry)
        objTry += dataTermTry   # add dataTerm (using finalStateTry) to objTry
        objTry += uTermTry      # add uTerm (using uTry) to objTry
        ztTry = copy.deepcopy(foo[1])
        if self.vf_type == 'LDDMM':
            self.updated_lineSearch = [uTry, ztTry, forwardStateInTry, forwardStateOutTry, atTry, AfftTry, dir, eps]
        if np.isnan(objTry):
            logging.info('Warning: nan in updateTry')
            return 1e500
        if (objRef == None) or (objTry < objRef):
            if self.vf_type == 'LDDMM':
                self.atTry = copy.deepcopy(atTry)
                self.AfftTry = copy.deepcopy(AfftTry)
            self.objTry = np.copy(objTry)
            self.uTry = copy.deepcopy(uTry)
            self.ztTry = copy.deepcopy(ztTry)                               # not a variable
            self.forwardStateInTry = copy.deepcopy(forwardStateInTry)       # not a variable
            self.forwardStateOutTry = copy.deepcopy(forwardStateOutTry)     # not a variable
        return objTry


    def addProd(self, dir1, dir2, beta):
        dir = Direction(vf_type=self.vf_type, nD=self.nD, nA=self.nA)
        if self.vf_type == 'LDDMM':
            dir.diff = [(dir1.diff[layer_D] + beta * dir2.diff[layer_D]) for layer_D in range(self.nD)]
            dir.aff = [(dir1.aff[layer_D] + beta * dir2.aff[layer_D]) for layer_D in range(self.nD)]
        dir.upred = [(dir1.upred[layer_A] + beta * dir2.upred[layer_A]) for layer_A in range(self.nA)]
        return dir


    def prod(self, dir1, beta):
        dir = Direction(vf_type=self.vf_type, nD=self.nD, nA=self.nA)
        if self.vf_type == 'LDDMM':
            dir.diff = [beta * dir1.diff[layer_D] for layer_D in range(self.nD)]
            dir.aff = [beta * dir1.aff[layer_D] for layer_D in range(self.nD)]
        dir.upred = [beta * dir1.upred[layer_A] for layer_A in range(self.nA)]
        return dir


    def copyDir(self, dir0):
        dir = Direction(vf_type=self.vf_type, nD=self.nD, nA=self.nA)
        if self.vf_type == 'LDDMM':
            dir.diff = copy.deepcopy(dir0.diff)
            dir.aff = copy.deepcopy(dir0.aff)
        dir.upred = copy.deepcopy(dir0.upred)
        return dir


    def randomDir(self):
        dirfoo = Direction(vf_type=self.vf_type, nD=self.nD, nA=self.nA)
        if self.vf_type == 'LDDMM':
            for layer_D in range(self.nD):
                dirfoo.diff[layer_D] = np.random.normal(size=self.at[layer_D].shape)
                dirfoo.aff[layer_D] = np.random.normal(size=self.Afft[layer_D].shape)
        for layer_A in range(self.nA):
            dirfoo.upred[layer_A] = np.random.normal(size=self.u[layer_A].shape)
        if self.model == 'classification':
            dirfoo.upred[-1][:, 0] = 0
        return dirfoo


    def dotProduct(self, g1, g2):   # used in cg
        if self.vf_type == 'LDDMM':
            res = np.zeros(len(g2))
            for layer_D in range(self.nD):
                for t in range(self.Tsize[layer_D]):
                    x = self.zt[layer_D][t, self.idx_diff, :]
                    gg = g1.diff[layer_D][t, :, :]
                    u = self.KparDiff[layer_D].applyK(x, gg)
                    uu = g1.aff[layer_D][t]
                    ll = 0
                    for gr in g2:
                        ggOld = gr.diff[layer_D][t, :, :]
                        res[ll] = res[ll] + np.multiply(ggOld, u).sum()
                        if self.affineDim[layer_D] > 0:
                            res[ll] += np.multiply(uu, gr.aff[layer_D][t]).sum() * self.coeffAff
                        ll = ll + 1
            pp = copy.deepcopy(g1.upred)
            for layer_A in range(self.nA):
                ll = 0
                for gr in g2:
                    res[ll] += (pp[layer_A] * gr.upred[layer_A]).sum() * self.coeffuPred
                    ll += 1
            return res


    def dotProduct_euclidean(self, g1, g2): # used in bfgs
        res = np.zeros(len(g2))
        if self.vf_type == 'LDDMM':
            for layer_D in range(self.nD):
                for t in range(self.Tsize[layer_D]):
                    gg = g1.diff[layer_D][t, :, :]
                    uu = g1.aff[layer_D][t]
                    ll = 0
                    for gr in g2:
                        ggOld = gr.diff[layer_D][t, :, :]
                        res[ll] = res[ll] + np.multiply(ggOld, gg).sum()
                        if self.affineDim[layer_D] > 0:
                            res[ll] += np.multiply(uu, gr.aff[layer_D][t]).sum() * self.coeffAff
                        ll = ll + 1
        pp = copy.deepcopy(g1.upred)
        for layer_A in range(self.nA):
            ll = 0
            for gr in g2:
                res[ll] += (pp[layer_A] * gr.upred[layer_A]).sum() * self.coeffuPred
                ll += 1
        return res


    def stopCondition(self):  # called by bfgs
        return False


    def startOfIteration(self):
        if self.reset:
            if self.vf_type == 'LDDMM':
                for K in self.KparDiff:
                    K.dtype = 'float64'


    def endOfProcedure(self):  # called by cg and bfgs
        self.endOfIteration(endP=True)

    ####################################################################################################################
    #   optimizationTools stop
    ####################################################################################################################

    ####################################################################################################################
    #   endOfIteration start
    ####################################################################################################################

    def endOfIteration(self, endP=False):  # called by cg and bfgs, and endOfProcedure
        if not endP:
            self.iter += 1

        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2

        if (self.iter % self.saveRate == 0 or endP):
            if endP:
                logging.info('-------------------------------endP-------------------------------')
                if self.vf_type == 'LDDMM':
                    self.obj, self.zt, self.forwardState_In, self.forwardState_Out = self.objectiveFunDef_LDDMM(self.at,
                                                                                                                self.Afft,
                                                                                                                withTrajectory=True)
            if self.model == 'regression':
                self.guTr = np.copy(self.forwardState_Out[-1][:,:self.dim_Y])
            elif self.model == 'classification':
                self.guTr = np.argmax(self.forwardState_Out[-1], axis=1)[:, np.newaxis]
            self.calculateErrors()
            self.concatenateResults()
            self.running_plot()

        if self.vf_type == 'LDDMM':
            for k,K in enumerate(self.KparDiff):
                K.dtype = self.Kdtype[k]

    ####################################################################################################################
    #   endOfIteration stop
    ####################################################################################################################

    ####################################################################################################################
    #   calculateErrors and concatenateResults start
    ####################################################################################################################

    def calculateErrors(self):
        if self.model == 'regression':
            if self.standardize_fit_y:
                self.train_err = (np.multiply(self.guTr - self.YTr, self.s0_y) ** 2).sum() / self.NTr
                self.train_results = np.multiply(self.guTr, self.s0_y) + self.mu0_y
            else:
                self.train_err = ((self.guTr - self.YTr) ** 2).sum() / self.NTr
                self.train_results = np.copy(self.guTr)
        elif self.model == 'classification':
            self.train_err = np.sum(np.not_equal(self.guTr, self.YTr) * self.wTr) / self.swTr
            self.train_results = np.copy(self.guTr)
        logging.info('Training Error {0:.8f}'.format(np.squeeze(self.train_err)))
        for layer_A in range(self.nA):
            U, self.u_sVals[layer_A], V = np.linalg.svd(self.u[layer_A])


    def concatenateResults(self):
        if len(self.trainError_array) == 0:
            self.trainError_array = np.copy(self.train_err[np.newaxis,np.newaxis])
            for layer_A in range(self.nA):
                self.u_sVals_array[layer_A] = np.copy(self.u_sVals[layer_A][:,np.newaxis])
            self.obj_array = np.copy(self.obj[np.newaxis,np.newaxis])
        else:
            self.trainError_array = np.concatenate((self.trainError_array, self.train_err[np.newaxis,np.newaxis]), axis=1)
            for layer_A in range(self.nA):
                self.u_sVals_array[layer_A] = np.concatenate((self.u_sVals_array[layer_A], self.u_sVals[layer_A][:,np.newaxis]), axis=1)
            self.obj_array = np.concatenate((self.obj_array, self.obj[np.newaxis,np.newaxis]), axis=1)

    ####################################################################################################################
    #   calculateErrors and concatenateResults stop
    ####################################################################################################################

from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
sys_path.append('../data')
import numpy as np
from data import dataSets
from diffeoLearn import DiffeoLearn


demo = 'regression'  # 'regression' or 'classification'

if demo == 'regression':
    # regression demo

    # read in train and test data
    typeData0 = 'rings'
    data = dataSets.DataConfiguration(typeData0=typeData0)

    # run model
    params = {'logging_flag':True, 'plot_flag':True, 'XTr_initFactor':0.0}
    f = DiffeoLearn(params)
    f.fit(data.XTr, data.YTr, typeData=typeData0, pykeops_cleanup=True)
    train_results = f.predict(data.XTr)
    test_results = f.predict(data.XTe)

    # calculate errors
    train_error = ((train_results - data.YTr) ** 2).sum() / data.YTr.shape[0]
    test_error = ((test_results - data.YTe) ** 2).sum() / data.YTe.shape[0]

elif demo == 'classification':
    # classification demo

    # read in train and test data
    typeData0 = 'tori3'
    data = dataSets.DataConfiguration(typeData0=typeData0)

    # run model
    params = {'model': 'classification', 'logging_flag': True, 'plot_flag': True}
    f = DiffeoLearn(params)
    f.fit(data.XTr, data.YTr, typeData=typeData0, pykeops_cleanup=True)
    train_results = f.predict(data.XTr)
    test_results = f.predict(data.XTe)

    # calculate errors
    nClasses = data.YTr.max() + 1
    nTr = np.array([(data.YTr == k).sum() for k in range(nClasses)])
    wTr = float(data.YTr.size) / (nTr[data.YTr[:, 0]] * nClasses)[:, np.newaxis]
    swTr = wTr.sum()
    train_error = np.sum(np.not_equal(train_results, data.YTr) * wTr) / swTr

    nTe = np.array([(data.YTe == k).sum() for k in range(nClasses)])
    wTe = float(data.YTe.size) / (nTr[data.YTe[:, 0]] * nClasses)[:, np.newaxis]
    swTe = wTe.sum()
    test_error = np.sum(np.not_equal(test_results, data.YTe) * wTe) / swTe

print('train error: ', train_error)
print('test error:   ', test_error)


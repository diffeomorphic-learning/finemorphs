from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
sys_path.append('../data')
import time
import logging
from base import loggingUtils
from datetime import timedelta
import numpy as np
from data import dataSets
from diffeoLearn import DiffeoLearn


#                   instances, attributes X (Y)
# concrete               1030,  8 (1)
# energy                  768,  8 (1)
# kin8nm                 8192,  8 (1)
# naval-propulsion      11934, 16 (1)
# power-plant            9568,  4 (1)
# protein-tertiary      45730,  9 (1)
# wine-quality-red       1599, 11 (1)
# yacht                   308,  6 (1)
# year_MSD             515345, 90 (1)
# airfoil                1503,  5 (1)


# regression experiments
experiments = []

experiments += [{'data': 'concrete',               'nRuns': 20,    'case': 0}]          # ADA
# experiments += [{'data': 'energy',                 'nRuns': 20,    'case': 0}]          #  |
# experiments += [{'data': 'kin8nm',                 'nRuns': 20,    'case': 0}]          #  |
# experiments += [{'data': 'naval-propulsion',       'nRuns': 20,    'case': 0}]          #  |
# experiments += [{'data': 'power-plant',            'nRuns': 20,    'case': 0}]          #  |
# experiments += [{'data': 'protein-tertiary',       'nRuns': 5,     'case': 0}]          #  |
# experiments += [{'data': 'wine-quality-red',       'nRuns': 20,    'case': 0}]          #  |
# experiments += [{'data': 'yacht',                  'nRuns': 20,    'case': 0}]          #  |
# experiments += [{'data': 'year_MSD',               'nRuns': 1,     'case': 1}]          #  |
# experiments += [{'data': 'concrete_gap',           'nRuns': 8,     'case': 0}]          #  |
# experiments += [{'data': 'energy_gap',             'nRuns': 8,     'case': 0}]          #  |
# experiments += [{'data': 'kin8nm_gap',             'nRuns': 8,     'case': 0}]          #  |
# experiments += [{'data': 'naval-propulsion_gap',   'nRuns': 16,    'case': 0}]          #  |
# experiments += [{'data': 'power-plant_gap',        'nRuns': 4,     'case': 0}]          #  |
# experiments += [{'data': 'protein-tertiary_gap',   'nRuns': 9,     'case': 0}]          #  |
# experiments += [{'data': 'wine-quality-red_gap',   'nRuns': 11,    'case': 0}]          #  |
# experiments += [{'data': 'yacht_gap',              'nRuns': 6,     'case': 0}]          #  |

# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 0}]          # ADA
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 2}]          #  |  (T=20)
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 3}]          #  |  (T=30)
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 4}]          #  |  (T=40)
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 5}]          # ADADA
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 6}]          # ADADADA
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 7}]          # ADADADADA
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 8}]          # ADADADADADA
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 9}]          # ADDA (incr. kernel widths)
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 10}]         # ADDDA         |
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 11}]         # ADDDDA        |
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 12}]         # ADDA (decr. kernel widths)
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 13}]         # ADDDA         |
# experiments += [{'data': 'airfoil',                'nRuns': 10,    'case': 14}]         # ADDDDA        |

# experiments += [{'data': 'concrete',               'nRuns': 20,    'case': 14}]        # ADDDDA (decr. kernel widths)
# experiments += [{'data': 'energy',                 'nRuns': 20,    'case': 14}]        # ADDDDA        |
# experiments += [{'data': 'kin8nm',                 'nRuns': 20,    'case': 14}]        # ADDDDA        |
# experiments += [{'data': 'power-plant',            'nRuns': 20,    'case': 14}]        # ADDDDA        |
# experiments += [{'data': 'wine-quality-red',       'nRuns': 20,    'case': 14}]        # ADDDDA        |
# experiments += [{'data': 'yacht',                  'nRuns': 20,    'case': 14}]        # ADDDDA        |
# experiments += [{'data': 'concrete_gap',           'nRuns': 8,     'case': 14}]        # ADDDDA        |
# experiments += [{'data': 'energy_gap',             'nRuns': 8,     'case': 14}]        # ADDDDA        |
# experiments += [{'data': 'kin8nm_gap',             'nRuns': 8,     'case': 14}]        # ADDDDA        |
# experiments += [{'data': 'power-plant_gap',        'nRuns': 4,     'case': 14}]        # ADDDDA        |
# experiments += [{'data': 'wine-quality-red_gap',   'nRuns': 11,    'case': 14}]        # ADDDDA        |
# experiments += [{'data': 'yacht_gap',              'nRuns': 6,     'case': 14}]        # ADDDDA        |


for expt in experiments:

    typeData0 = expt['data']
    nRuns = expt['nRuns']
    case = expt['case']

    if case == 0: params = {}
    if case == 1: params = {'trainSubset_flag':True, 'layerOutDim':[10,None,None]}
    if case == 2: params = {'Tsize':20}
    if case == 3: params = {'Tsize':30}
    if case == 4: params = {'Tsize':40}
    if case == 5: params = {'layers':['A','D','A','D','A']}
    if case == 6: params = {'layers':['A','D','A','D','A','D','A']}
    if case == 7: params = {'layers':['A','D','A','D','A','D','A','D','A']}
    if case == 8: params = {'layers':['A','D','A','D','A','D','A','D','A','D','A']}
    if case == 9: params = {'layers':['A','D','D','A'], 'kernelSigma':[0.33,0.66]}
    if case == 10: params = {'layers':['A','D','D','D','A'], 'kernelSigma':[0.25,0.50,0.75]}
    if case == 11: params = {'layers':['A','D','D','D','D','A'], 'kernelSigma':[0.2,0.4,0.6,0.8]}
    if case == 12: params = {'layers':['A','D','D','A'], 'kernelSigma':[0.66,0.33]}
    if case == 13: params = {'layers':['A','D','D','D','A'], 'kernelSigma':[0.75,0.5,0.25]}
    if case == 14: params = {'layers':['A','D','D','D','D','A'], 'kernelSigma':[0.8,0.6,0.4,0.2]}

    logging_flag = True
    loggingFile = typeData0+'_case_'+str(case)+'_info'
    base_params = {'dtype':'float32', 'logging_flag':logging_flag, 'loggingFile':loggingFile}

    parameters = {**params, **base_params}

    train_errors = np.zeros((1, nRuns))
    test_errors = np.zeros((1, nRuns))
    times = np.zeros((1, nRuns))

    pykeops_cleanup = True

    f = DiffeoLearn(parameters)
    for run in range(nRuns):

        # read in data
        data = dataSets.DataConfiguration(typeData0=typeData0, run=run)
        typeData = typeData0 + '_' + str(run) + '_case_' + str(case)
        print('\n{0:s}\n'.format(typeData))

        # run model
        startTime = time.time()
        f.fit(data.XTr, data.YTr, typeData=typeData, pykeops_cleanup=pykeops_cleanup)
        train_results = f.predict(data.XTr)
        test_results = f.predict(data.XTe)
        endTime = time.time()

        # calculate errors and times
        train_errors[0, run] = ((train_results - data.YTr) ** 2).sum() / data.YTr.shape[0]
        test_errors[0, run] = ((test_results - data.YTe) ** 2).sum() / data.YTe.shape[0]
        times[0, run] = endTime - startTime
        print('    elapsed time: {:0>8}'.format(str(timedelta(seconds=times[0, run]))))

        pykeops_cleanup = False

    loggingUtils.setup_default_logging(f.outputDir, fileName=loggingFile, stdOutput=True, mode='a')  # 'a' = write without overwriting
    logging.info('-' * 121)

    # analyze results
    logging.info('\n{0}  case {1}'.format(typeData0,case))
    logging.info('avg RMSE of train errors:   {0: .8f}  +/- {1: .8f}'.format(np.mean(np.sqrt(train_errors)),
                                                                             np.std(np.sqrt(train_errors)) / np.sqrt(nRuns)))
    logging.info('avg RMSE of test errors:    {0: .8f}  +/- {1: .8f}'.format(np.mean(np.sqrt(test_errors)),
                                                                             np.std(np.sqrt(test_errors)) / np.sqrt(nRuns)))
    logging.info('total run time:              {:0>8}'.format(str(timedelta(seconds=np.sum(times)))))
    logging.info('avg run time:                {:0>8}\n'.format(str(timedelta(seconds=np.mean(times)))))

    # close logging file
    logger = logging.getLogger()
    for h in logger.handlers:
        logger.removeHandler(h)

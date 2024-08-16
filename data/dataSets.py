import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def ringUniform(n, a=0., b=1., d=2):
    m = np.zeros(d)
    S = np.eye(d)
    X = np.random.multivariate_normal(m, S, size=n)
    nrm = np.sqrt((X**2).sum(axis=1))
    X /= nrm[:, np.newaxis]
    r = np.random.uniform(0,1, size=n)
    r = (a**d + r*(b**d-a**d))**(1/d)
    return X*r[:, np.newaxis]


class DataConfiguration:
    def __init__(self, typeData0=None, run=None):
        if typeData0 is None:
            self.XTr = None
            self.YTr = None
            self.XTe = None
            self.YTe = None
        else:
            self.init(typeData0, run)

    def init(self, typeData0, run):
        XTr = None
        YTr = None
        XTe = None
        YTe = None

                                                        #  instances, attributes X (Y)
        if typeData0 in ('concrete',                    #       1030,  8 (1)
                         'energy',                      #        768,  8 (1)
                         'kin8nm',                      #       8192,  8 (1)
                         'naval-propulsion',            #      11934, 16 (1)
                         'power-plant',                 #       9568,  4 (1)
                         'protein-tertiary',            #      45730,  9 (1)
                         'wine-quality-red',            #       1599, 11 (1)
                         'yacht',                       #        308,  6 (1)
                         'concrete_gap',
                         'energy_gap',
                         'kin8nm_gap',
                         'naval-propulsion_gap',
                         'power-plant_gap',
                         'protein-tertiary_gap',
                         'wine-quality-red_gap',
                         'yacht_gap'):
            dataDir = './data/UCI/' + typeData0 + '/data/'
            index_features = list((pd.read_csv(dataDir + 'index_features.txt', sep=" ", header=None)).to_numpy(dtype='int').ravel())
            index_target = list((pd.read_csv(dataDir + 'index_target.txt', sep=" ", header=None)).to_numpy(dtype='int').ravel())
            if typeData0 in ('kin8nm','naval-propulsion','kin8nm_gap','naval-propulsion_gap'):
                all_data = pd.read_fwf(dataDir + 'data.txt', header=None)
            elif typeData0 in ('protein-tertiary','wine-quality-red','yacht','protein-tertiary_gap','wine-quality-red_gap','yacht_gap'):
                all_data = pd.read_csv(dataDir + 'data.txt', sep="\s+", header=None)
            else:
                all_data = pd.read_csv(dataDir + 'data.txt', sep="\t", header=None)
            index_train_data = list((pd.read_csv(dataDir + 'index_train_' + str(run) + '.txt', sep=" ", header=None)).to_numpy(dtype='int').ravel())
            index_test_data = list((pd.read_csv(dataDir + 'index_test_' + str(run) + '.txt', sep=" ", header=None)).to_numpy(dtype='int').ravel())
            XTr = all_data.iloc[index_train_data, index_features].to_numpy(dtype='float64')
            YTr = all_data.iloc[index_train_data, index_target].to_numpy(dtype='float64')
            XTe = all_data.iloc[index_test_data, index_features].to_numpy(dtype='float64')
            YTe = all_data.iloc[index_test_data, index_target].to_numpy(dtype='float64')
        elif typeData0 == 'year_MSD':   # instances = 515345, attributes = 90 (1)
            df = pd.read_table('./data/UCI/year_prediction_MSD/YearPredictionMSD.txt', sep=',', header=None)
            x = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                            80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]].to_numpy(dtype='float64')
            y = df.iloc[:, [0]].to_numpy(dtype='float64')
            XTr = x[:463715,:]
            YTr = y[:463715,:]
            XTe = x[463715:,:]
            YTe = y[463715:,:]
        elif typeData0 == 'airfoil':   # instances = 1503, attributes = 5 (1)
            dataDir = './data/UCI/airfoil/'
            df = pd.read_table(dataDir + 'airfoil_self_noise.dat', names=('A', 'B', 'C', 'D', 'E', 'F'))
            x_orig = df.iloc[:, [0, 1, 2, 3, 4]].to_numpy(dtype='float64')
            y_orig = df.iloc[:, [5]].to_numpy(dtype='float64')
            x = np.copy(x_orig)
            y = np.copy(y_orig)
            train_size = 0.9
            XTr, XTe, YTr, YTe = train_test_split(x, y, train_size=train_size, random_state=run+1, shuffle=True)
        elif typeData0 == 'rings':
            NTr = 2000
            NTe = 10000
            d = 2
            x = ringUniform(NTr + NTe, d=d)
            nrm = np.sqrt((x ** 2).sum(axis=1))
            y = -np.array(np.cos(4 * nrm))[:, np.newaxis]
            XTr = x[:NTr,:]
            YTr = y[:NTr,:]
            XTe = x[NTr:,:]
            YTe = y[NTr:,:]
        elif typeData0 == 'tori3':
            NTr = 2000
            NTe = 2000
            NTr0 = NTr // 2
            NTe0 = NTe // 2
            d = 3
            c = 1.
            h = 0.25 * c
            XTr = 0.05 * np.random.randn(2 * NTr0, d)
            XTe = 0.05 * np.random.randn(2 * NTe0, d)
            YTr = np.ones((2 * NTr0, 1), dtype=int)
            YTr[NTr0:2 * NTr0] = 0
            YTe = np.ones((2 * NTe0, 1), dtype=int)
            YTe[NTe0:2 * NTe0] = 0
            t = 2 * np.pi * np.random.rand(NTr0)
            s = 2 * np.pi * np.random.rand(NTr0)
            XTr[0:NTr0, 0] += c * np.cos(t) + h * np.cos(s)
            XTr[0:NTr0, 1] += c * np.sin(t) + h * np.cos(s)
            XTr[0:NTr0, 2] += h * np.sin(s)
            XTr[NTr0:2 * NTr0, 0] += h * np.sin(s)
            XTr[NTr0:2 * NTr0, 1] += c + c * np.cos(t) + h * np.cos(s)
            XTr[NTr0:2 * NTr0, 2] += c * np.sin(t) + h * np.cos(s)
            XTr[:, 3:d] += 1. * np.random.randn(2 * NTr0, d - 3)
            t = 2 * np.pi * np.random.rand(NTe0)
            s = 2 * np.pi * np.random.rand(NTe0)
            XTe[0:NTe0, 0] += c * np.cos(t) + h * np.cos(s)
            XTe[0:NTe0, 1] += c * np.sin(t) + h * np.cos(s)
            XTe[0:NTe0, 2] += h * np.sin(s)
            XTe[NTe0:2 * NTe0, 0] += h * np.sin(s)
            XTe[NTe0:2 * NTe0, 1] += c + c * np.cos(t) + h * np.cos(s)
            XTe[NTe0:2 * NTe0, 2] += c * np.sin(t) + h * np.cos(s)
            XTe[:, 3:d] += 1. * np.random.randn(2 * NTe0, d - 3)

        self.XTr = np.copy(XTr)
        self.YTr = np.copy(YTr)
        self.XTe = None
        self.YTe = None
        if XTe is not None:
            self.XTe = np.copy(XTe)
            self.YTe = np.copy(YTe)


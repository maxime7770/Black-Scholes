#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import keras
import keras.backend as K
#
# !pip install py_vollib
#
from sklearn.model_selection import ParameterGrid
from py_vollib import black_scholes_merton as bsm
from progressbar import ProgressBar
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import uniform
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split

# S (spot price)
# gamma


def thisS(q):
    return gamma.ppf(q, a=100, scale=1)
# K (strike price)
# uniform (lower = 50, upper = 200)


def thisK(q):
    return uniform.ppf(q, 50, 200)
# (interest rate)
# uniform (lower = 0.01, upper = 0.18)


def thisR(q):
    return uniform.ppf(q, 0.01, 0.18)
# D (dividend)
# uniform (lower = 0.01, upper = 0.18)


def thisD(q):
    return uniform.ppf(q, 0.01, 0.18)


# t (time-to-maturity)
# t will be 3, 6, 9, 12 months for all examples (0.25, 0.5, 0.75, 1 year)
# sigma (volatility)
# beta (add small amount so volatility cannot be zero)
def thisSigma(q):
    return (beta.ppf(q, a=2, b=5) + 0.001)


num_increment = 5
percentiles = pd.Series(np.linspace(0, 0.99, num_increment))
S = percentiles.apply(thisS)
K = percentiles.apply(thisK)
q = percentiles.apply(thisD)
t = np.array([.25, .5, .75, 1])
r = percentiles.apply(thisR)
sigma = percentiles.apply(thisSigma)
param_grid = {'S': S, 'K': K, 'q': q, 't': t, 'r': r, 'sigma': sigma}
grid = ParameterGrid(param_grid)
18
pbar = ProgressBar()
sparseDF = pd.DataFrame()
prices = []
for params in pbar(grid):
    prices.append(bsm.black_scholes_merton(flag='p', S=params['S'], K=params['K'],
                                           q=params['q'], t=params['t'], r=params['r'], sigma=params['sigma']))
    sparseDF = sparseDF.append(pd.Series(params), ignore_index=True)
# swap price to first column
sparseDF['price'] = prices
# output to csv
sparseDF.to_csv('dataSparse.csv', index=False)
print(sparseDF.head())
print(sparseDF.tail())

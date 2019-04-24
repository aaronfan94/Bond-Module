#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:06:05 2019

@author: AaronFan
"""

import numpy as np
import pandas as pd

freqToYears = {'D': 252.0,
               'M': 12.0,
               'Q': 4.0,
               'S': 2.0,
               'A': 1.0}


class PCATerm:
    '''
    Construct PCA induced term structure
    '''

    def __init__(self, n_comp=None, tol=1e-4, freq='A'):
        self.n_comp = n_comp
        self.__tol = tol
        self.freq = freqToYears[freq]

    def chg_currYields(self, Y):
        # check if Y is of the same dimension
        if len(Y) != len(self.terms):
            raise TypeError('Yields not of the same length.')
        # store the new term structure
        self.currYields = np.array(Y)
        '''
        fit linear interpolation function
        '''
        from scipy.interpolate import interp1d
        if 0.0 not in self.__yrs:
            X = np.append(0.0, self.__yrs)
            Y = np.append(0.0, self.currYields)
        else:
            X = self.__yrs
            Y = self.currYields
        self.__fitted_currYields = interp1d(X, Y)

    def get_yield(self, t, which_yield='current'):

        if which_yield == 'current':
            return np.float(self.__fitted_currYields(t))

        # create loadings in case it doesn't exist
        if hasattr(self, 'loadings') is False:
            loadings = self.get_loadings()
        else:
            loadings = self.loadings

        # interpolation between loadings
        res = np.zeros(loadings.shape[0])
        for i in range(loadings.shape[0]):
            res[i] = np.float(self.__fitted_loadings[i](t))

        # interpolate between current loadings
        last_yield = np.float(self.__fitted_currYields(t))

        # the res is the interpolated y_chg
        if isinstance(which_yield, int):
            return res.dot(self.pcs[which_yield]) + last_yield
        elif which_yield == 'random':
            idx_rand = np.random.choice(range(len(self.pcs)))
            return res.dot(self.pcs[idx_rand]) + last_yield
        else:
            raise TypeError('Either index or use random simulation.')

    def fit(self, y_chg, currYields=None):
        if isinstance(y_chg, pd.DataFrame) is False:
            raise TypeError('y must be a pd.DataFrame.')
        self.trueCurve = np.array(y_chg)
        self.terms = y_chg.columns
        self.currYields = np.array(currYields)
        self.__yrs = np.array([float(i) / self.freq for i in self.terms])
        '''
        fitting the PCA to the change of yields
        '''
        # Principle component transformation
        y_chg_var = np.array(y_chg.corr(method='pearson'))
        w, v = np.linalg.eig(y_chg_var)

        self.eig_value = w
        self.var_explained = np.cumsum(w) / np.sum(w)
        self.eig_vector = v

        if self.n_comp is None:
            # record the change of explainary power
            chg_exp = np.diff(self.var_explained)
            try:
                self.n_comp = np.where(chg_exp < self.__tol)[0][0]
            except IndexError:
                # if the changes are not small enough, do not cut
                self.n_comp = len(v)

        # get the corresponding eigen vectors
        Q = v.T[:self.n_comp]
        self.pcs = self.trueCurve.dot(Q.T)

        return self

    def get_loadings(self, interp=True, **kargs):
        '''
        because the pcs are orthonormal by construction
        the loadings are
        pc = y * Q -> y = pc * Q'
        '''
        Q = self.eig_vector
        self.loadings = pd.DataFrame(Q.T[:self.n_comp], columns=self.terms)
        self.fitted = self.pcs.dot(np.array(self.loadings))
        self.__fitted_loadings = [[]] * self.n_comp

        if interp:
            '''
            fit each loadings by methods of linear or spline
            '''
            import copy

            # get interp method
            try:
                method = kargs['method']
            except KeyError:
                method = 'linear'

            # interpolate
            if method == 'linear':
                from scipy.interpolate import interp1d
                # store loading interpolate functions
                for i in range(self.loadings.shape[0]):
                    # use linear interp
                    f = interp1d(x=self.__yrs,
                                 y=np.array(self.loadings.iloc[i, :]))
                    self.__fitted_loadings[i] = copy.copy(f)

            elif method == 'spline':
                from scipy.interpolate import UnivariateSpline
                # get the order from input otherwise use 3
                try:
                    fit_order = kargs['k']
                except KeyError:
                    fit_order = 3
                # store loading interpolate functions
                for i in range(self.loadings.shape[0]):
                    # use cubic spline
                    f = UnivariateSpline(x=self.__yrs,
                                         y=np.array(self.loadings.iloc[i, :]),
                                         k=fit_order)
                    self.__fitted_loadings[i] = copy.copy(f)
            else:
                raise TypeError('Interpolation Method not Implemented.')

        return self.loadings


if __name__ == '__main__':

    inpath = '/Users/AaronFan/Desktop/Risk Management/midterm/'

    # read term structure
    df = pd.read_csv(inpath + 'TermStructure.csv').set_index('Unnamed: 0')
    df = df.dropna(axis=1, how='any') / 100
    # construct raw changes
    y_chg = df.diff().dropna(axis=0, how='any')

    m = PCATerm(n_comp=4, freq='D').fit(y_chg)
    print(m.get_loadings(method='spline', k=5))
    m.chg_currYields(np.zeros(20))
    X = np.array([float(i) / 252 for i in m.terms])

    import matplotlib.pyplot as plt
    idx = -100
    plt.plot(m.fitted[idx], label='Fitted')
    plt.plot(np.array(y_chg)[idx], label='Original')
    plt.plot([m.get_yield(i, which_yield=idx) for i in X], label='Spline')
    plt.legend()
    plt.show()

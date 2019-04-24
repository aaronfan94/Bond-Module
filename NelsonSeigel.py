#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:06:05 2019

@author: AaronFan
"""
import numpy as np

from AF.Helpers.StatsHelpers import reg

'''
Term Structure fitting with Nelson Seigel
'''


def NSfunc(TimeExp, params):
    z1 = params[0]
    z2 = params[1]
    z3 = params[2]
    eta = params[3]

    k2 = (1 - np.exp(-eta * TimeExp)) / (eta * TimeExp)
    k3 = k2 - np.exp(-eta * TimeExp)

    return z1 + k2 * z2 + k3 * z3


class NSTerm:

    # initialize with fixed shape parameters
    def __init__(self, eta=None, params=None):
        self.eta = eta
        # allow user to define NS
        if params is not None:
            self.z1 = params[0]
            self.z2 = params[1]
            self.z3 = params[2]

    # return parameters
    def get_param(self):
        return {'z1': self.z1,
                'z2': self.z2,
                'z3': self.z3,
                'eta': self.eta}

    # get yields from term structure
    def get_yield(self, t):
        try:
            # single time point
            return NSfunc(t, [self.z1, self.z2, self.z3, self.eta])
        except TypeError:
            # multiple time points
            return [NSfunc(i, [self.z1, self.z2, self.z3, self.eta]) for i in t]

    # fitting term structure with given shape:
    def fit(self, y, TimeExp):
        # record the true curve
        self.trueCurve = {'TimeToMaturity': TimeExp,
                          'Yields': y}
        # if shape is set
        if self.eta is not None:

            import pandas as pd

            # construct regressors
            TimeExp = np.array(TimeExp)
            k2 = (1 - np.exp(-self.eta * TimeExp)) / (self.eta * TimeExp)
            k3 = k2 - np.exp(-self.eta * TimeExp)
            X = pd.DataFrame({'x1': k2, 'x2': k3})

            #  estimate coefficients and record
            q = reg(y, X).params
            self.z1 = q[0]
            self.z2 = q[1]
            self.z3 = q[2]

        else:
            from scipy.optimize import minimize

            def obj_min(params):
                ssr = 0
                for i in range(len(TimeExp)):
                    t = TimeExp[i]
                    ssr = ssr + (NSfunc(t, params) - y[i])**2
                return ssr

            res = minimize(obj_min, [0.0, 0.0, 0.0, 0.5])
            if res.success is False:
                print('Optimization Failed.')

            self.z1 = res.x[0]
            self.z2 = res.x[1]
            self.z3 = res.x[2]
            self.eta = res.x[3]

        # recorded fitted line
        self.fitted = {'TimeToMaturity': TimeExp,
                       'Yields': self.get_yield(TimeExp)}

        return self


if __name__ == '__main__':

    y = [0.01, 0.011, 0.015, 0.02, 0.025, 0.03, 0.05]
    TimeExp = [1 / 12, 3 / 12, 6 / 12, 1.0, 2.0, 3.0, 5.0]

    print(NSTerm(eta=0.01).fit(y, TimeExp).get_param())
    m = NSTerm().fit(y, TimeExp)
    print(m.__dict__)

    import matplotlib.pyplot as plt
    plt.plot(m.trueCurve['Yields'], label='true')
    plt.plot(m.fitted['Yields'], label='fitted')
    plt.legend()
    plt.show()

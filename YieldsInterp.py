#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:06:05 2019

@author: AaronFan
"""
import numpy as np


def get_yields(curve, t, **kargs):
    from AF.AnalyticalLib.NelsonSeigel import NSTerm
    from AF.AnalyticalLib.PCATermStruct import PCATerm

    if isinstance(curve, NSTerm):
        return curve.get_yield(t)

    elif isinstance(curve, PCATerm):
        '''
        PCA gives the choice of which pc movement to price
        '''
        return curve.get_yield(t, which_yield=kargs['which_curve'])

    elif isinstance(curve, ZeroCurve):
        return curve.get_yield(t)


class ZeroCurve:
    '''
    this is a generic zero coupon curve
    '''

    def __init__(self, terms, yields, **kargs):
        from scipy.interpolate import interp1d
        self.terms = np.array(terms)
        self.yields = np.array(yields)
        self.yieldFunc = interp1d(terms, yields, **kargs)

    def get_yield(self, t):
        return np.float(self.yieldFunc(t))


if __name__ == '__main__':

    y = [0.01, 0.011, 0.015, 0.02, 0.025, 0.03, 0.05]
    TimeExp = [1 / 12, 3 / 12, 6 / 12, 1.0, 2.0, 3.0, 5.0]

    zc = ZeroCurve(TimeExp, y)

    print(get_yields(zc, 4))

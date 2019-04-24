#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:06:05 2019

@author: AaronFan
"""

import numpy as np
from AF.Helpers.UniHelpers import seq, pos
from AF.AnalyticalLib.YieldsInterp import get_yields


freqToYears = {'M': 1.0 / 12,
               'Q': 3.0 / 12,
               'S': 6.0 / 12,
               'A': 1.0}


def ZCBpricing(y, t):
    return np.exp(-y * t)


class bond:
    '''
    Bond instrument for pricing
    '''

    def __init__(self, maturity, freq, **kargs):
        self.maturity = maturity
        self.freq = freq
        if self.freq is not 'ZCB':
            self.couponRate = kargs['couponRate']
            self.couponDays = seq(0, maturity, freqToYears[freq])[1:]

    def get_YTM(self):
        from scipy.optimize import minimize

        # pricing control
        maturity = self.maturity - self.valueDay / 252
        couponDays = pos(self.couponDays - self.valueDay / 252)

        def obj_min(y):
            res = ZCBpricing(y, maturity)
            if self.freq is not 'ZCB':
                for t in couponDays:
                    res += self.couponRate * freqToYears[self.freq] * ZCBpricing(y, t)
            return (res - self.currPrice)**2

        m = minimize(obj_min, 0.05)
        if m.success is False:
            print('Optimization Failed.')

        self.YTM = m.x[0]

        return self.YTM

    def price(self, curve, valueDay=0, which_curve=-1):
        '''
        if pricing with PC augmented curve, user can choose
        which pc movement to price with
        default to -1 which is the last observation
        '''

        self.valueDay = valueDay
        maturity = self.maturity - valueDay / 252
        tmp_y = get_yields(curve, maturity, which_curve=which_curve)
        res = ZCBpricing(tmp_y, maturity)

        if self.freq is not 'ZCB':
            # use the futre coupon days
            couponDays = pos(self.couponDays - valueDay / 252)
            for t in couponDays:
                tmp_y = get_yields(curve, t, which_curve=which_curve)
                res += self.couponRate * freqToYears[self.freq] * ZCBpricing(tmp_y, t)

        # record current price
        self.currPrice = res
        return res


if __name__ == '__main__':
    from AF.AnalyticalLib.NelsonSeigel import NSTerm
    from AF.AnalyticalLib.PCATermStruct import PCATerm
    import pandas as pd
    inpath = '/Users/AaronFan/Desktop/Risk Management/midterm/'

    # read term structure
    df = pd.read_csv(inpath + 'TermStructure.csv').set_index('Unnamed: 0')
    df = df.dropna(axis=1, how='any') / 100
    TimExp = np.array([int(i) / 252.0 for i in np.array(df.columns)])
    # construct raw changes
    y_chg = df.diff().dropna(axis=0, how='any')

    pc_curve = PCATerm(n_comp=5, freq='D').fit(y_chg)
    pc_curve.chg_currYields(df.iloc[-1, :])
    loadings = pc_curve.get_loadings()

    ns_curve = NSTerm().fit(df.iloc[-1, :], TimExp)

    cb = bond(7, 'S', couponRate=0.15)
    print(cb.price(pc_curve, which_curve=1))
    print(cb.get_YTM())
    print(cb.price(pc_curve, which_curve=10))
    print(cb.price(pc_curve, which_curve=50))
    print(cb.price(ns_curve))

    cb = bond(7, 'ZCB')
    print(cb.price(pc_curve, which_curve=1))
    print(cb.price(pc_curve, which_curve=10))
    print(cb.price(pc_curve, which_curve=50))
    print(cb.price(ns_curve))

## This code is written by Marco Chierici <chierici@fbk.eu> and Alessandro Zandona' <zandona@fbk.eu>.
## Based on code previously written by Davide Albanese.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import numpy as np
import mlpy
from scaling import norm_l2
import performance as perf
from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing

def svmlin_t(x, y, LOSS, PENALTY, DUAL, scaling, list_C, cv_k, cv_p, logReg=False, quiet=False):
    
    if not quiet:
        print 'Scaling: %s' %scaling
    idx = mlpy.cv_random(n=x.shape[0], k=cv_k, p=cv_p, strat=y)
    
    # average MCC on test, avg MCC on train, avg error on test, avg error on train
    AMCC_ts, AMCC_tr, AERR_ts, AERR_tr = [], [], [], []
    # vectors containing CV metrics
    mcc_ts = np.zeros(cv_k, dtype=np.float)
    mcc_tr = np.zeros(cv_k, dtype=np.float)
    err_ts = np.zeros(cv_k, dtype=np.float)
    err_tr = np.zeros(cv_k, dtype=np.float)

    for C in list_C:
        for i, (idx_tr, idx_ts) in enumerate(idx):
            x_tr, x_ts = x[idx_tr], x[idx_ts]
            y_tr, y_ts = y[idx_tr], y[idx_ts]

            # centering and standardization
            if scaling == 'norm_l2':
                x_tr, m_tr, r_tr = norm_l2(x_tr) 
                x_ts, _, _ = norm_l2(x_ts, m_tr, r_tr) 
            elif scaling == 'std':
                scaler = preprocessing.StandardScaler(copy=False)
                x_tr = scaler.fit_transform(x_tr)
                x_ts = scaler.transform(x_ts)
            elif scaling == 'minmax':
                scaler = preprocessing.MinMaxScaler(feature_range=(-1,1), copy=False)
                x_tr = scaler.fit_transform(x_tr)
                x_ts = scaler.transform(x_ts)
            elif scaling == 'minmax0':
                scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=False)
                x_tr = scaler.fit_transform(x_tr)
                x_ts = scaler.transform(x_ts)

            if logReg:
                clf = linear_model.LogisticRegression(C=C, penalty=PENALTY, dual=DUAL, class_weight='auto', tol=1e-2)
            else:
                clf = svm.LinearSVC(C=C, loss=LOSS, penalty=PENALTY, dual=DUAL, class_weight='auto', tol=0.01)

            # train
            clf.fit(x_tr, y_tr)
            # predict on test, train
            ypts = clf.predict(x_ts)
            yptr = clf.predict(x_tr)

            # MCC on test, train
            mcc_ts[i] = perf.KCCC_discrete(y_ts, ypts)
            mcc_tr[i] = perf.KCCC_discrete(y_tr, yptr)

            # error on test, train
            err_ts[i] = perf.error(y_ts, ypts)
            err_tr[i] = perf.error(y_tr, yptr)

        AMCC_ts.append(np.mean(mcc_ts))
        AMCC_tr.append(np.mean(mcc_tr))
        AERR_ts.append(np.mean(err_ts))
        AERR_tr.append(np.mean(err_tr))
        
        if not quiet:
            print "C: %f -> MCC %.3f, test error %.3f (train MCC %.3f, train error %.3f)" % (C, AMCC_ts[-1], AERR_ts[-1], AMCC_tr[-1], AERR_tr[-1])
    
    # best C maximizes AMCC_ts
    bestC_idx = np.argmax(AMCC_ts)
    # return train/test MCC and Error
    return list_C[bestC_idx], AMCC_ts[bestC_idx], AERR_ts[bestC_idx], AMCC_tr[bestC_idx], AERR_tr[bestC_idx]


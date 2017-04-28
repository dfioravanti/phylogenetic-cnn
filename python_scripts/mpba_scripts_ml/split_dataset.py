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
import sys
import os.path

def split_tr_ts(x, y, p, seed=0):
    idx = mlpy.cv_random(x.shape[0], 10, p, strat=y, seed=seed)
    x_tr, x_ts = x[idx[0][0]], x[idx[0][1]]
    y_tr, y_ts = y[idx[0][0]], y[idx[0][1]]

    return x_tr, y_tr, x_ts, y_ts


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "Usage: %prog data.txt labels.txt test_size outprefix [random_seed]"
        sys.exit(3)

    # file to split
    datafile = sys.argv[1]
    # labels
    labelfile = sys.argv[2]
    # test size
    p = np.int( sys.argv[3] )
    # output prefix ( _tr.txt, _tr.lab, _ts.txt, _ts.lab will be appended)
    outprefix = sys.argv[4]
    # random seed
    if len(sys.argv) == 6:
        random_seed = np.int( sys.argv[5] )
    else:
        random_seed = 0

    data = np.loadtxt(datafile, delimiter='\t', dtype=str)
    # header
    h = data[0]
    # data (with sample column)
    x = data[1:]
    # labels
    y = np.loadtxt(labelfile, dtype=np.int)

    x_tr, y_tr, x_ts, y_ts = split_tr_ts(x, y, p, seed=random_seed)

    np.savetxt( outprefix + '_tr.txt', np.concatenate( (h.reshape(1,-1), x_tr), axis=0 ), fmt='%s', delimiter='\t' )
    np.savetxt( outprefix + '_ts.txt', np.concatenate( (h.reshape(1,-1), x_ts), axis=0 ), fmt='%s', delimiter='\t' )
    np.savetxt( outprefix + '_tr.lab', y_tr, fmt='%d')
    np.savetxt( outprefix + '_ts.lab', y_ts, fmt='%d')



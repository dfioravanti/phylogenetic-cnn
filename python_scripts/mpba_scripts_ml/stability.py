## This code is written by Davide Albanese <albanese@fbk.eu>

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
import csv
import os.path
import sys
import argparse

parser = argparse.ArgumentParser(description='Compute the Canberra stability indicator given a RANKING file.')
parser.add_argument('RANKFILE', type=str, help='Ranking datafile')
parser.add_argument('METRICSFILE', type=str, help='Metrics datafile (FSTEPS will be read from the 1st column')
parser.add_argument('-f', '--force', action='store_true', help='Force overwriting of output files.')

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
RANKFILE = args.RANKFILE
METRICSFILE = args.METRICSFILE
overwrite = args.force
OUTFILE = '_'.join(METRICSFILE.split('_')[:-1] + ['stability.txt'])

if os.path.exists(OUTFILE) and not overwrite:
    print "Existing output file %s" % (OUTFILE)
    sys.exit(2)

stabilityf = open(OUTFILE, 'w')
stability_w = csv.writer(stabilityf, delimiter='\t', lineterminator='\n')
stability_w.writerow(["STEP", "STABILITY"])

RANKING = np.loadtxt(RANKFILE, delimiter='\t', dtype=np.int)
tmp = np.loadtxt(METRICSFILE, delimiter='\t', skiprows=1, dtype=str)
FSTEPS = tmp[1:,0].astype(np.int)

# check dimensions
if RANKING.shape[1] < np.max(FSTEPS):
    FSTEPS = np.append( FSTEPS[FSTEPS<RANKING.shape[1]] , RANKING.shape[1] )

STABILITY = []
PR = np.argsort( RANKING )
for ss in FSTEPS:
    STABILITY.append( mlpy.canberra_stability(PR, ss) )

for j, s in enumerate(FSTEPS):
    stability_w.writerow( [s, STABILITY[j]] )

stabilityf.close()

import numpy as np
import csv
import sys

def extract_feats(datafile, rankedfile, nfeat, outfile):
    print locals()
    # table with feats abundances 
    data = np.loadtxt(datafile, delimiter = '\t', dtype = str)
    # feats abundances (no names of samples, no header)
    data_ab = data[1:,1:].astype(np.float)

    rank = np.loadtxt(rankedfile, delimiter = '\t', skiprows = 1, dtype = str)
    feats = rank[:, 1]
    top_feats = feats[0:nfeat]

    print top_feats.shape
    # extract top features from table with abundances of all features
    idx = []
    for i in range(0, nfeat):
        if top_feats[i] in data[0,:].tolist():
            idx.append(data[0,:].tolist().index(top_feats[i]))
        else:
            print top_feats[i]

    # considering samples names in the new table
    idx = [0] + idx
    sel_feats = data[:, idx]

    # write new table
    outw = open(outfile, 'w')
    writer = csv.writer(outw, delimiter = '\t', lineterminator = '\n')
    for i in range(0, len(sel_feats[:,0])):
        writer.writerow(sel_feats[i,:])

    outw.close()


if __name__ == "__main__":
    if not len(sys.argv) == 5:
        print "Usage: %prog data.txt rankingfile nfeat outdata.txt"
        sys.exit(1)

    # file with all feats abundances (where selected feats have to be picked from)
    datafile = sys.argv[1]
    # file with ranked features
    rankedfile = sys.argv[2]
    # number of feat to extract
    nfeat = int(sys.argv[3])
    # file with abundances of the only selected features
    outfile = sys.argv[4]

    extract_feats(datafile, rankedfile, nfeat, outfile)


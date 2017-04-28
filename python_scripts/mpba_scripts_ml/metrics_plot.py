## This code is written by Alessandro Zandona' <zandona@fbk.eu> and 
## Marco Chierici <chierici@fbk.eu>.

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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, os.path
import sys
import re
import time
import argparse
import pandas as pd
import csv

def lfchange_plot(featList=None, numFeat=10, title='Cases vs control', greengenes=False):
    import seaborn as sns
    # Feature list
    fList = pd.read_csv(featList, sep='\t', index_col=0)
    
    if greengenes:
        taxon = ['P', 'C', 'O', 'F', 'G', 'S', ''] 
        taxa_lev = {}
        for t, k in enumerate(['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']):
            taxa_lev[k] = taxon[t]
        axLab = []
        for j in range(0, numFeat+1):
            features = fList['FEATURE_NAME'].iloc[j].split(';')
            lastTaxa = features[-1]
            if (len(lastTaxa)>3 and lastTaxa != 'Other'):
                axLab.append(lastTaxa)
            else:
                for i in range(0, len(features)-1)[::-1]:
                    if (len(features[i])>3 and features[i]!= 'Other'):
                        axLab.append(features[i]+' (unsp. ' + taxa_lev[features[i][:3]] + ')')
                        break
    else:
        axLab = fList['FEATURE_NAME']

    plt.close()
    plt.figure()
    lf_df = pd.DataFrame({'Index': axLab[:numFeat][::-1], 'Value': fList['LOG2_FOLD_CHANGE'][:numFeat][::-1]})
    lfp = lf_df.plot('Index', 'Value', kind='barh', title=title, legend=False)
    sns.despine()
    lfp.set_ylabel('Feature')
    lfp.set_xlabel('Log2(fold change)')
    plt.tight_layout()

    outFile = featList.replace('_featurelist.txt', '_logFoldChange.pdf')
    plt.savefig(outFile)
    plt.close()
    plt.clf()

def topfeats_plot(featList=None, numFeat=10, abTable=None, labFile=None, outFld=None, greengenes=False, log=False):
    import seaborn as sns
    # Abundance table 
    abundTable = pd.read_csv(abTable, sep='\t', index_col=0)
    # Feature list
    featList = np.loadtxt(featList, delimiter='\t', dtype=str, skiprows=1)
    # File with class labels 
    labFile = np.loadtxt(labFile, dtype=np.str)
    # Labels as data frame
    lab_df = pd.DataFrame(labFile, columns=["Classes"], index=abundTable.index)
    # Consider the top features
    topFeats = featList[:numFeat,1]
    dataTop = abundTable[topFeats]
    # Concatenate abundance table and class labels
    df = pd.concat([dataTop, lab_df], axis=1)
    if greengenes:
        taxon = ['P', 'C', 'O', 'F', 'G', 'S', ''] 
        taxa_lev = {}
        for t, k in enumerate(['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']):
                taxa_lev[k] = taxon[t]
        xlabels = []
        for j in range(0, len(topFeats)):
            features = df.columns[j].split(';')
            lastTaxa = features[-1]
            if (len(lastTaxa)>3 and lastTaxa != 'Other'):
                xlabels.append(lastTaxa)
            else:
                for i in range(0, len(features)-1)[::-1]:
                    if (len(features[i])>3 and features[i]!= 'Other'):
                        xlabels.append(features[i]+' (unsp. ' + taxa_lev[features[i][:3]] + ')')
                        break
        df.columns = xlabels + ['Classes']
    else:
        xlabels = df.columns[:numFeat]

    dfm = pd.melt(df, id_vars=["Classes"], value_name="counts", var_name="Feature")

    plt.close()
    plt.figure()
    sns.set_style('ticks')

    g = sns.factorplot("Feature", "counts", "Classes", dfm, kind="box", x_order=xlabels, legend_out=False)
    if log:
        g.fig.get_axes()[0].set_yscale('log')
        g.set_axis_labels("Feature", "Log10( relative abund )")
        outFile_pdf = os.path.join(outFld, os.path.basename(abTable).replace('.txt', '_topFeat_abundance_log.pdf'))
        outFile_png = os.path.join(outFld, os.path.basename(abTable).replace('.txt', '_topFeat_abundance_log.png'))
    else:
        g.set_axis_labels("Feature", "Relative abundance")
        outFile_pdf = os.path.join(outFld, os.path.basename(abTable).replace('.txt', '_topFeat_abundance.pdf'))
        outFile_png = os.path.join(outFld, os.path.basename(abTable).replace('.txt', '_topFeat_abundance.png'))

    g.set_xticklabels(xlabels,rotation=90, size='xx-small')
    plt.title('Top ranked features abundance')
    plt.tight_layout()
    plt.savefig(outFile_pdf, dpi=400)

    plt.savefig(outFile_png, dpi=400)
    plt.close()
    plt.clf()

def most_abund(abTable=None, labFile=None, outFile=None, nfeat=10, greengenes=False):
    import seaborn as sns
    # read table with features abundances 
    abundTable = np.loadtxt(abTable, delimiter='\t', dtype=str)
    # convert abundTable into a dataframe
    abundTable_df = pd.read_csv(abTable, sep = '\t', index_col=0)
    # read labels file
    labF = np.loadtxt(labFile, dtype=str)
    # label file as dataframe
    lab_df = pd.DataFrame(labF, columns=["Classes"])
    # compute the abundance of each feature across all the samples
    abund = abundTable[1:,1:].astype(np.float)
    # abundance in the first class
    abund_cl1 = abund[ labF==pd.unique(lab_df['Classes'])[0], : ]
    abund_cl2 = abund[ labF==pd.unique(lab_df['Classes'])[1], : ]
    # sum abundance over samples from first class
    colsum_cl1 = abund_cl1.sum(axis=0)
    # sum abundance over samples from second class
    colsum_cl2 = abund_cl2.sum(axis=0)

    # indexes of features sorted by abundance - first class
    idx_sort_cl1 = np.argsort(colsum_cl1)[::-1]
    # indexes of features sorted by abundance - second class
    idx_sort_cl2 = np.argsort(colsum_cl2)[::-1]

    # extract names of the most abundant features - first class
    topabund_feat_cl1 = abundTable[ 0, idx_sort_cl1[:nfeat]+1 ]
    # extract names of the most abundant features - second class
    topabund_feat_cl2 = abundTable[ 0,idx_sort_cl2[:nfeat]+1 ]


    # table with samples from first class and most abundant features in first class
    topabund_cl1_tmp = abund[ :, idx_sort_cl1[:nfeat] ]
    topabund_cl1 = topabund_cl1_tmp[labF==pd.unique(lab_df['Classes'])[0] ,:]
    # table with samples from first class and most abundant features in second class
    topabund_cl2_tmp = abund[ : ,idx_sort_cl2[:nfeat] ]
    topabund_cl2 = topabund_cl2_tmp[labF==pd.unique(lab_df['Classes'])[1] ,:]

    # dataframe with only the most abundant features in the first class
    topabund_df_cl1 = pd.DataFrame(topabund_cl1, columns=topabund_feat_cl1)
    # dataframe with only the most abundant features in the second class
    topabund_df_cl2 = pd.DataFrame(topabund_cl2, columns=topabund_feat_cl2)

    if greengenes:
        taxon = ['P', 'C', 'O', 'F', 'G', 'S', '']
        taxa_lev = {}

        for t, k in enumerate(['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']):
            taxa_lev[k] = taxon[t]

        xlabels_cl1 = []
        for j in range(0, nfeat):
            features = topabund_df_cl1.columns[j].split(';')
            lastTaxa = features[-1]
            if (len(lastTaxa)>3 and lastTaxa != 'Other'):
                xlabels_cl1.append(lastTaxa)
            else:
                for i in range(0, len(features)-1)[::-1]:
                    if (len(features[i])>3 and features[i]!= 'Other'):
                        xlabels_cl1.append(features[i]+' (unsp. ' + taxa_lev[features[i][:3]] + ')')
                        break

        xlabels_cl2 = []
        for j in range(0, nfeat):
            features = topabund_df_cl2.columns[j].split(';')
            lastTaxa = features[-1]
            if (len(lastTaxa)>3 and lastTaxa != 'Other'):
                xlabels_cl2.append(lastTaxa)
            else:
                for i in range(0, len(features)-1)[::-1]:
                    if (len(features[i])>3 and features[i]!= 'Other'):
                        xlabels_cl2.append(features[i]+' (unsp. ' + taxa_lev[features[i][:3]] + ')')
                        break

        topabund_df_cl1.columns = xlabels_cl1
        topabund_df_cl2.columns = xlabels_cl2
    else:
        xlabels_cl1 = topabund_df_cl1.columns
        xlabels_cl2 = topabund_df_cl2.columns


    # melt dataset
    dfm_cl1 = pd.melt(topabund_df_cl1, value_name="counts", var_name="Feature")
    dfm_cl2 = pd.melt(topabund_df_cl2, value_name="counts", var_name="Feature")

    plt.close()
    plt.figure()
    # plot class 1
    sns.set_style('ticks')
    g = sns.barplot("Feature", "counts",data=dfm_cl1, x_order=xlabels_cl1)

    g.set_xticklabels(xlabels_cl1,rotation=90 )
    sns.despine(left=False)
    plt.ylabel("Relative abundance")
    plt.title("Most abundant features in " + pd.unique(lab_df['Classes'])[0])
    plt.tight_layout()
    plt.savefig(outFile + '_' + pd.unique(lab_df['Classes'])[0] + '.pdf')

    # plot class 2
    g = sns.barplot("Feature", "counts", data=dfm_cl2, x_order=xlabels_cl2 )

    g.set_xticklabels(xlabels_cl2,rotation=90)

    plt.ylabel("Relative abundance")
    plt.title("Most abundant features in " + pd.unique(lab_df['Classes'])[1])
    plt.tight_layout()

    plt.savefig(outFile + '_'+ pd.unique(lab_df['Classes'])[1] + '.pdf')
    plt.close()
    plt.clf()

def mcc_distrib(mccFile=None, metricsFile=None, outFile=None):
    import seaborn as sns
    # read dataset with metric values over 50 CV cycles as dataframe
    df = pd.read_csv(mccFile, delimiter='\t', header=None) 
    # number of features leading to the best MCC
    reader = csv.reader(open(metricsFile, 'r'), delimiter='\t')
    topF = np.int(reader.next()[1])
    # feature steps
    fs = np.loadtxt(metricsFile, delimiter='\t', skiprows=2, usecols=(0,), dtype=np.int)
    df.columns=fs
    # create a molten dataframe
    dfm = pd.melt(df, var_name="Feature step", value_name="MCC")
    plt.close()
    plt.figure()
    sns.set_style('ticks')
    # plot boxplots for MCC values over the 10x5-fold CV, for each feature step
    col_bxp = len(fs)*["orange"]
    topF_idx = fs.tolist().index(topF)
    col_bxp[topF_idx] = "darkgreen"
    bxp = sns.boxplot(df, names=df.columns, color=col_bxp, alpha=0.8)
    # set orientation and size of labels on X axis
    bxp.set_xticklabels(df.columns,rotation=45, size='x-small')
    # set name of x axis
    bxp.set_xlabel('Feature steps')
    # set name of y axis
    bxp.set_ylabel('MCC')
    sns.despine(trim=True)
    # title
    plt.title("MCC distribution over CV")
    plt.tight_layout()
    # save figure
    if outFile is not None:
        outF = outFile
    else: 
        outFile = re.sub('metrics.txt', 'MCC_distrib.pdf', metricsFile)
    plt.savefig(outFile)
    plt.close()
    plt.clf()

def rankplot(rankFile=None, bordaFile=None, Nfeat=10, positions=20, outFile=None):
    RANKING = np.loadtxt(rankFile, delimiter='\t', dtype=np.int)
    BORDA = np.loadtxt(bordaFile, delimiter='\t', dtype=str, usecols=(0,1,2,), skiprows=1)
    fid, fname, fpos = BORDA[:Nfeat,0].astype(np.int), BORDA[:Nfeat,1], BORDA[:Nfeat,2].astype(np.float)

    plt.close('all')
    # create Nfeat subplots sharing both axes
    f, ax = plt.subplots(Nfeat, sharex=True, sharey=True)
    f.suptitle("Marker rank position")
    
    for i,a in enumerate(ax):
        cnt = np.sum(RANKING[:, :positions]==fid[i], axis=0)
        a.bar(range(positions), cnt, color='b')
        a.axvline(fpos[i], 0, color='r')
        a.set_ylabel("Feat_%d" % (fid[i]), rotation='horizontal', horizontalalignment='right', size='small')
        #a.set_ylabel(fname[i], rotation='horizontal', horizontalalignment='right', size='xx-small')
        #if i==0:
        #    a.set_title("Marker rank position")
        if i==(Nfeat-1):
            a.set_xlabel("Position", size='small')

    f.subplots_adjust(hspace=0)
    # hide xticklabels for all axes except the most bottom one
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    # hide yticklabels for all axes
    plt.setp([a.get_yticklabels() for a in f.axes], visible=False)
    plt.tight_layout()
    if outFile is not None:
        outF = outFile
    else:
       # outF = os.path.join(os.path.dirname(rankFile), 'Feats_position_distribution.pdf')
        outF = re.sub('featurelist.txt', 'feats_pos_distribution.pdf', bordaFile)
    
    plt.savefig(outF)
    plt.close()
    plt.clf()

def metplot(normFile=None, RRFile=None, RLFile=None, metric='MCC', format_plot='pdf', title='', outprefix=None):
    """Returns the plot of a specific metric values across the CV cycles
    
    :Parameters:
       normFile : string
                file path of table with all metrics computed by ML
       RRFile : string
                None or file path of table with all metrics computed by ML with random ranking approach
       RLFile : string
                None or file path of table with all metrics computed by ML with random labels approach       
       metric : string
                metric to plot
       format_plot : string
                format to use for the plot 
       outprefix : string
                output directory of final plot 

    """
    import seaborn as sns
    plt.close('all')
    sns.set_style("darkgrid")
    sns.set_context("paper")
    D = {}
    if not normFile is None:
        D['normFile'] = ['k', '-', 'Normal', normFile]
    if not RRFile is None:
        D['RRFile'] = ['blue', '--', 'RR', RRFile]
    if not RLFile is None:
        D['RLFile'] = ['green', '-.', 'RL', RLFile]

    if D == {}:
        raise IOError("Error: specify at least one input file!")

    if outprefix is not None:
        out_fld = outprefix
    else:
        out_fld = os.path.dirname(D[D.keys()[0]][3])

    #plt.figure(1)
    ax = plt.subplot(111)
    plt.title( metric + ' plot ' + '\n' + title)

    for i in range(len(D)):
        plotFile = D.keys()[i]
        all_metrics, ymet, idx = init_plot(D[D.keys()[i]][3], metric)
        fsteps = np.array(all_metrics[1:,0], dtype = np.int)
        plt.errorbar(all_metrics[1:, 0], all_metrics[1:, idx].astype(np.float), yerr=ymet.astype(np.float), color=D[plotFile][0], linestyle = D[plotFile][1], label = D[plotFile][2])
        
    myTime = time.time()
    outfile = re.sub('(?i)metrics', metric + '_' + str(len(D)) +'plots', os.path.basename(D[D.keys()[0]][3]) )
    if (len(D)==1):
        outfile = re.sub('plots', 'plot', outfile)
    uniqueStr = str(myTime) + '.' + format_plot
    outfile = re.sub('txt', uniqueStr, outfile)

    plt.legend(loc=4)
    plt.xlabel("Number of features")
    plt.ylabel("Average " + metric + " with 95% c.i.")
    ax.set_xscale('log')
    plt.xlim((0.5, (np.max(fsteps)+30)))
    plt.ylim((-0.4, 1))
    #plt.ylim((-1, 1.1))
    plt.savefig(os.path.join(out_fld, outfile), format=format_plot)
    plt.close()
    plt.clf()

def metplot_sns(normFile=None, RRFile=None, RLFile=None, metricsFile=None, metric='MCC', format_plot='pdf', title='', outFld=None):
    """Returns the plot of a specific metric values across the CV cycles
    
    :Parameters:
       normFile : string
                file path of table with values of selected metric computed by ML over the CV cycles
       RRFile : string
                None or file path of table with values of selected metric computed by ML over the CV cycles with random ranking approach
       RLFile : string
                None or file path of table with values of selected metric computed by ML over the CV cycles with random labels approach       
       metricsFile: string
	        file path of table with all metrics computed by ML for each features step 
       metric : string
                metric to plot
       format_plot : string
                format to use for the plot 
       outFld : string
       		name of folder for final plot 

    """
    import seaborn as sns 
    plt.close('all')
    D = {}
    if not normFile is None:
        D['normFile'] = [ 'Normal', normFile]
    if not RRFile is None:
        D['RRFile'] = ['RR', RRFile]
    if not RLFile is None:
        D['RLFile'] = ['RL', RLFile]
	
    if outFld is not None:
        outFolder = outFld
    else:
	outFolder = os.path.dirname(D[D.keys()[0]][1])
	
    myTime = str(time.time()).split('.')[0]
    outFile = re.sub('(?i)metrics', metric + '_' + str(len(D)) +'plots_seaborn', os.path.basename(D[D.keys()[0]][1]) )
    if (len(D)==1):
        outFile = re.sub('plots', 'plot', outFile)
    uniqueStr = str(myTime) + '.' + format_plot
    outFile = re.sub('txt', uniqueStr, outFile)

 
    fs = np.loadtxt(metricsFile, delimiter='\t', dtype=np.int, usecols=(0,), skiprows=2)
    tmp = np.loadtxt(normFile, delimiter='\t', dtype=np.float)
    metrics = pd.DataFrame(tmp, columns=fs)
    
    tmp = np.loadtxt(RLFile, delimiter='\t', dtype=np.float)
    RLmetrics = pd.DataFrame(tmp, columns=fs)
    tmp = np.loadtxt(RRFile, delimiter='\t', dtype=np.float)
    RRmetrics = pd.DataFrame(tmp, columns=fs)
    # add factor column with label type
    metrics['Labels'] = metrics.shape[0] * ['Normal']
    RLmetrics['Labels'] = RLmetrics.shape[0] * ['RL']
    RRmetrics['Labels'] = RRmetrics.shape[0] * ['RR']
    # concatenate the 3 dataframes along the 0 axis
    df = pd.concat([metrics, RLmetrics, RRmetrics], axis=0)
    dfm = pd.melt(df, id_vars="Labels", var_name="feature step", value_name="MCC")
    
    sns.set_style("ticks")
    sns.set_context("paper")
    pal = dict(Normal="#333333", RR="#6495ED", RL="#2E8B57")
    g = sns.factorplot("feature step", metric, hue="Labels", data=dfm, palette=pal, ci=95, n_boot=1000, kind="point", legend=False)
    # set logx
    g.fig.get_axes()[0].set_xscale('log')
    sns.despine()
    g.set_xticklabels(df.columns,rotation=45, size='x-small')
    plt.legend(loc=4)
    plt.ylim((-1, 1.1))
    plt.title(metric + ' plot ' + '\n' + title)
    plt.tight_layout()
    plt.savefig(os.path.join(outFolder, outFile), format=format_plot)
    plt.close()
    plt.clf()


def init_plot(metricsFile, metric):
    """Returns the plot of a specific metric values across the CV cycles
    
    :Parameters:
       metricsFile : string
                file path of table with all metrics computed by ML
       metric : string
                metric to plot

    :Returns:
       all_metrics : matrix 
             matrix with all metrics computed by ML for each CV cycle
       ymet : 2d array_like float 
             confidence intervals 
       idx : int
             column of metrics matrix with values of selected metric
    """
    all_metrics =  np.loadtxt(metricsFile, delimiter='\t', skiprows=1, dtype = str)
    # select metric to plot
    met_names = all_metrics[0,:].tolist()
    idx = met_names.index(metric)
    # array with metric values for all the CV cycles
    met = all_metrics[1:,idx]
    # confidence intervals
    met_min_idx = met_names.index(metric + '_MIN')
    met_max_idx = met_names.index(metric + '_MAX')

    ymet = all_metrics[1:, [met_min_idx, met_max_idx ]].T
    ymet[0] = all_metrics[1:, idx].astype(np.float) - ymet[0].astype(np.float)
    ymet[1] = ymet[1].astype(np.float) - all_metrics[1:, idx].astype(np.float)

    return all_metrics, ymet, idx



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_lfc = subparsers.add_parser('lfchange')
    parser_lfc.add_argument('featList',  type=str, help='File path of ranked features list')
    parser_lfc.add_argument('--numFeat', type=np.int, help='Number of features to plot (default=%(default)s)', default=10)
    parser_lfc.add_argument('--title', type=str, help='Title of plot (default: %(default)s)', default='Cases vs control')
    parser_lfc.add_argument('--greengenes', action='store_true', help='Variables: Greengenes taxonomy')
    parser_lfc.set_defaults(name='lfchange', func=lfchange_plot)

    parser_topf = subparsers.add_parser('topfeats')
    parser_topf.add_argument('featList', type=str, help='File path of ranked features list')
    parser_topf.add_argument('--numFeat', type=np.int, help='Number of features to plot (default=%(default)s)', default=10)
    parser_topf.add_argument('abTable', type=str, help='File path of the dataset with features abundance')
    parser_topf.add_argument('labFile', type=str, help='File path of class labels')
    parser_topf.add_argument('outFld', type=str, help='Output folder')
    parser_topf.add_argument('--greengenes', action='store_true', help='Variables: Greengenes taxonomy')
    parser_topf.add_argument('--log', action='store_true', help='Plot features abundance in log scale')
    parser_topf.set_defaults(name='topfeats', func=topfeats_plot)

    parser_mab = subparsers.add_parser('mostabund')
    parser_mab.add_argument('abTable', type=str, help='File path of dataset to plot')
    parser_mab.add_argument('labFile', type=str, help='File path of class labels')
    parser_mab.add_argument('outFile', type=str, help='Output file')
    parser_mab.add_argument('--nfeat', dest='nfeat', type=np.int, help='Number of features to plot (default=%(default)s)', default=10)
    parser_mab.add_argument('--greengenes', action='store_true', help='Variables: Greengenes taxonomy')
    parser_mab.set_defaults(name='mostabund', func=most_abund) 

    parser_rp = subparsers.add_parser('rankplot')
    parser_rp.add_argument('rankFile', type=str, help='ranking.csv file')
    parser_rp.add_argument('bordaFile', type=str, help='featurelist.txt file')
    parser_rp.add_argument('-n', '--nfeat', dest='Nfeat', type=int, default=5, help='Number of top features to plot (default: %(default)s)')
    parser_rp.add_argument('-p', '--positions', dest='positions', type=int, default=20, help='Number of positions to stop at in the ranking file (default: %(default)s)')
    parser_rp.set_defaults(name='rankplot', func=rankplot)

    parser_mp = subparsers.add_parser('metplot')
    parser_mp.add_argument('--normFile', type=str, help='file path of table with all metrics computed by ML')
    parser_mp.add_argument('--RRFile', type=str, help='file path of table with all metrics computed by ML with random ranking')
    parser_mp.add_argument('--RLFile', type=str, help='file path of table with all metrics computed by ML with random labels')
    parser_mp.add_argument('--metric', type=str, help='metric to plot (default=%(default)s)', default='MCC')
    parser_mp.add_argument('--plot_format', type=str, help='Output file format (default=%(default)s)', default='pdf')
    parser_mp.add_argument('--title', type=str, help='Plot title', default='')
    parser_mp.add_argument('--outprefix', type=str, help='Output file prefix')
    parser_mp.set_defaults(name='metplot', func=metplot)

    parser_mpsns = subparsers.add_parser('metplot_seaborn')
    parser_mpsns.add_argument('--metricsFile', type=str, help='file path of table with all metrics computed by ML for each features step')
    parser_mpsns.add_argument('--normFile', type=str, help='file path of table with values of selected metric computed by ML over the CV cycles')
    parser_mpsns.add_argument('--RRFile', type=str, help='file path of table with values of selected metric computed by ML over the CV cycles with random ranking')
    parser_mpsns.add_argument('--RLFile', type=str, help='file path of table with values of selected metric computed by ML over the CV cycles with random labels')
    parser_mpsns.add_argument('--metric', type=str, help='metric to plot (default=%(default)s)', default='MCC')
    parser_mpsns.add_argument('--plot_format', type=str, help='Output file format (default=%(default)s)', default='pdf')
    parser_mpsns.add_argument('--title', type=str, help='Plot title', default='')
    parser_mpsns.add_argument('--outFld', type=str, help='Output folder')
    parser_mpsns.set_defaults(name='metplot_sns', func=metplot_sns)
    
    parser_mcd = subparsers.add_parser('mcc_distrib')
    parser_mcd.add_argument('--mccFile', type=str, help='file path of table with values of MCC computed by ML over the CV cycles')
    parser_mcd.add_argument('--metricsFile', type=str, help='file path of table with all metrics computed by ML for each features step')
    parser_mcd.add_argument('--outFile', type=str, help='Output file')
    parser_mcd.set_defaults(name='mcc_distribution', func=mcc_distrib)

    args = parser.parse_args()

    if args.name == 'rankplot':
        args.func(args.rankFile, args.bordaFile, args.Nfeat, args.positions)
    elif args.name == 'metplot':
        args.func(normFile=args.normFile, RRFile=args.RRFile, RLFile=args.RLFile, metric=args.metric, format_plot=args.plot_format, title=args.title, outprefix=args.outprefix)
    elif args.name == 'metplot_sns':
        args.func(normFile=args.normFile, RRFile=args.RRFile, RLFile=args.RLFile, metricsFile=args.metricsFile, metric=args.metric, format_plot=args.plot_format, title=args.title, outFld=args.outFld)
    elif args.name == 'mcc_distribution':
        args.func(mccFile=args.mccFile, metricsFile=args.metricsFile, outFile=args.outFile)
    elif args.name == 'mostabund':
        args.func(abTable=args.abTable, labFile=args.labFile, outFile=args.outFile, nfeat=args.nfeat, greengenes=args.greengenes)
    elif args.name == 'topfeats':
        args.func(featList=args.featList, numFeat=args.numFeat, abTable=args.abTable, labFile=args.labFile, outFld=args.outFld, greengenes=args.greengenes, log=args.log)
    elif args.name == 'lfchange':
        args.func(featList=args.featList, numFeat=args.numFeat, title=args.title, greengenes=args.greengenes)


    #dataDir = '/nfs/mlbio/data/opbg2014/ML_results_v2/FEC_H_B_H/l2r_l1loss_svc_dual'
    #rankFile = os.path.join(dataDir, 'FEC_H_B_H_genus_unass_p20_l2r_l1loss_svc_dual_SVM_std_ranking.csv')
    #rankFile = '/home/chierici/work/machine_learning/results/seqc2_A/SEQC_NB_AG1_G_log2_80-20_tr_l2r_l2loss_svc_dual_SVM_std_ranking.csv'
    #bordaFile = os.path.join(dataDir, 'FEC_H_B_H_genus_unass_p20_l2r_l1loss_svc_dual_SVM_std_featurelist.txt')
    #bordaFile = '/home/chierici/work/machine_learning/results/seqc2_A/SEQC_NB_AG1_G_log2_80-20_tr_l2r_l2loss_svc_dual_SVM_std_featurelist.txt'
    #rankplot(rankFile, bordaFile, Nfeat=6)


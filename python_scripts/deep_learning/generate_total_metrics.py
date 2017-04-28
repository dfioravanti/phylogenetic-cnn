import os
import numpy as np
import sys

def main():
    output_folder = sys.argv[1]
    # print output_folder
    f_steps = [ x for x in os.listdir(output_folder+'/') if x.endswith('_Features') and os.path.isdir(output_folder+'/'+x)]
    f_steps.sort()
    first  = True
    output = []
    for fld in f_steps:
        fld = output_folder+'/'+fld+'/'
        metrics_file = [x for x in os.listdir(fld) if x.endswith('_metrics.txt')][0]
        metrics = np.loadtxt(fld + metrics_file, dtype=str, delimiter='\t')
        if first:
            first = False
            output.append(metrics[0])
        output.append(metrics[1])
    output = np.array(output)
    np.savetxt(output_folder+'/total_metrics.txt', output, delimiter='\t', fmt='%s')


if __name__ == '__main__':
    main()
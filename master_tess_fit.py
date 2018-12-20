#
#
#  master_tess_fit.py
#  
#  Copyright 2018 Edward Bryant <phrvdf@monju.astro.warwick.ac.uk>
#  
#  Master code to batch run "tess_quickfit.py" for various input LC files
#  
#  

import argparse
import os
from time import time

def ParseArgs():
    '''
    Function to parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, nargs='*', help="Name of LC data file")
    parser.add_argument('-lk', '--lightkurve', action='store_true', help="Add this to use lightkurve to flatten the flux time series")
    parser.add_argument('-s', '--save', action='store_true', help="Add this to save the output plot instead of displaying")
    return parser.parse_args()

if __name__ == '__main__':
    
    args = ParseArgs()
    
    for fn in args.fn:
        time1 = time()
        if args.lightkurve:
            if args.save:
                os.system('python tess_quickfit.py --fn {} -lk -s'.format(fn))
            else:
                os.system('python tess_quickfit.py --fn {} -lk'.format(fn))
			    
        else:
            if args.save:
                os.system('python tess_quickfit.py --fn {} -s'.format(fn))
            else:
                os.system('python tess_quickfit.py --fn {}'.format(fn))
        time2 = time()
        print('Total time = {:.4f} s'.format(time2-time1))

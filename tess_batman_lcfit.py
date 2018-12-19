#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  tess_batman_lcfit.py
#  
#  29th November 2018
#  Edward Bryant <phrvdf@monju.astro.warwick.ac.uk>
#  
#  
#  

import argparse
import numpy as np
import batman as bm
from scipy.stats import chisquare, sem
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from time import time as TIME
from astropy.io import fits as pyfits
from lightkurve import TessLightCurve
import pandas

def ParseArgs():
    '''
    Function to parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str, help="Name of LC data file")
    parser.add_argument('--pipe', type=str, default='spoc', help="Pipeline used to reduce data (spoc or qlp)")
    parser.add_argument('-lk', '--lightkurve', action='store_true', help="Add this to use lightkurve to flatten the flux time series")
    parser.add_argument('--wl', default=1301, type=int)
    return parser.parse_args()
    
def lc_min(params, phase, flux, err):
	'''
	Function which calculates Chi2 value for a given set of input parameters.
	Function to be minimized to find best fit parameters
	'''
	
	#Define the system parameters for the batman LC model
	pm = bm.TransitParams()
	
#	pm.t0 = params['t0'].value   #Time of transit centre
	pm.t0 = 0.   #Time of transit centre
	pm.per = 1.                  #Orbital period = 1 as phase folded
	pm.rp = params['rp'].value   #Ratio of planet to stellar radius
	pm.a = params['a'].value             #Semi-major axis (units of stellar radius)
	pm.inc = params['inc'] .value          #Orbital Inclination [deg]
	pm.ecc = 0.                  #Orbital eccentricity (fix circular orbits)
	pm.w = 90.                   #Longitude of periastron [deg] (unimportant as circular orbits)
	pm.u = [0.1, 0.3]            #Stellar LD coefficients
	pm.limb_dark="quadratic"     #LD model
	
	#Initialize the batman LC model and compute model LC
	m = bm.TransitModel(pm, phase)
	f_model = m.light_curve(pm)
	residuals = (flux - f_model)**2/err**2
	return residuals

def lc_bin(time, flux, bin_width):
	'''
	Function to bin the data into bins of a given width. time and bin_width 
	must have the same units
	'''
	
	edges = np.arange(np.min(time), np.max(time), bin_width)
	dig = np.digitize(time, edges)
	time_binned = (edges[1:] + edges[:-1]) / 2
	flux_binned = np.array([np.nan if len(flux[dig == i]) == 0 else flux[dig == i].mean() for i in range(1, len(edges))])
	err_binned = np.array([np.nan if len(flux[dig == i]) == 0 else sem(flux[dig == i]) for i in range(1, len(edges))])
	time_bin = time_binned[~np.isnan(err_binned)]
	err_bin = err_binned[~np.isnan(err_binned)]
	flux_bin = flux_binned[~np.isnan(err_binned)]	
	
	return time_bin, flux_bin, err_bin	

def tess_LC_dataload_spoc(file_name):
	'''Loads TESS LC data for a given object and uses quality flags etc. to remove bad points 
		'''
	#Load FITS file
	hdul = pyfits.open(file_name)               #Read in FITS file
	hdr = hdul[0].header                  		#Primary FITS header
	
	TIC = hdr['TICID']							#Loads TIC ID number from the fits header
	Tmag = hdr['TESSMAG']						#Loads TESS mag from fits header
	Rs = hdr['RADIUS']							#Loads stellar radius [solar radius] from fits header
	sector = hdr['SECTOR']						#Loads observing sector data was obtained in			
	
	DATA_WHOLE = hdul[1].data             		#Extracts whole data

	#Extract desired columns
	time = DATA_WHOLE['TIME']			  		#Time [BJD - 2457000]
	#time_corr = DATA_WHOLE['TIMECORR']    		#Time correction: time - time_corr gives light arrival time at spacecraft

	FLUX = DATA_WHOLE['PDCSAP_FLUX']  		#PDC corrected flux from target star
	FLUX_ERR = DATA_WHOLE['PDCSAP_FLUX_ERR']#Error in PDC corrected flux
	
	#Load Quality flags in to remove flagged data points
	flags = DATA_WHOLE['QUALITY']
	flag_indices = np.where(flags > 0)

	flux_flagremoved = np.delete(FLUX, flag_indices)
	fluxerr_flagremoved = np.delete(FLUX_ERR, flag_indices)
	time_flagremoved = np.delete(time, flag_indices)

	#Remove time points during central gap
	null_indices = np.where(np.isnan(time_flagremoved))
	time_nullremoved = np.delete(time_flagremoved, null_indices)
	flux_nullremoved = np.delete(flux_flagremoved, null_indices)
	fluxerr_nullremoved = np.delete(fluxerr_flagremoved, null_indices)
	
	return time_nullremoved, flux_nullremoved, fluxerr_nullremoved, TIC, Tmag, Rs
	
def tess_LC_dataload_qlp(file_name):
	
	#Load FITS file
	hdul = pyfits.open(file_name)           		#Read in FITS file
	hdr = hdul[0].header                  		#Primary FITS header
	DATA_WHOLE = hdul[1].data             		#Extracts whole data

	#Extract desired columns
	time = DATA_WHOLE['TIME']			  		#Time [BJD - 2457000]
	FLUX = DATA_WHOLE['SAP_FLUX']
	
	#Load Quality flags in to remove flagged data points
	flags = DATA_WHOLE['QUALITY']
	flag_indices = np.where(flags > 0)

	flux_flagremoved = np.delete(FLUX, flag_indices)
	time_flagremoved = np.delete(time, flag_indices)

	#Remove time points during central gap
	null_indices = np.where(np.isnan(time_flagremoved))
	time_nullremoved = np.delete(time_flagremoved, null_indices)
	flux_nullremoved = np.delete(flux_flagremoved, null_indices)
	
	return time_nullremoved, flux_nullremoved

def lc_fit_spoc(time, flux, err, period, epoc, tdur, objid, Tmag, Rs):
    '''
    Function to fit a batman model to an input lc datafile to find the best 
    fit system parameters
    '''

    time0 = TIME()
    
    phase = ((time - epoc)/period)%1  #Convert time values in to phases
    phase = np.array([p-1 if p > 0.8 else p for p in phase], dtype=float)
    
    p_fit = phase[phase < 0.2]  #Crop phase and flux arrays to only contain values
    f_fit = flux[phase < 0.2]   #in range (-0.2 ,  0.2)
    e_fit = err[phase < 0.2]
    
    transit_indices = np.where(np.abs(p_fit * period) <= tdur / (2 * 24))	#Array indices of all phase/flux values during the transit
    FLUX_OOT = np.delete(f_fit, transit_indices)				#"Out Of Transit" flux values

    median = np.median(FLUX_OOT)										#median of all out-of-transit flux values

    f_fit /= median
    e_fit /= median
    
    params=Parameters()         #Parameter instance to hold fit parameters
    params.add('rp', value=0.05, min=0., max=1.)    #Planet:Star radius ratio
    params.add('a', value=10., min=0., max=100.)    #Semi-major axis
    params.add('inc', value=89., min=89., max=90.)  #Orbital inclination
#    params.add('t0', value=0.0, min=-0.1, max=0.1) #Transit centre time
    
    res = minimize(lc_min, params, args=(p_fit, f_fit, e_fit), method='leastsq') #perform minimization
    chi2 = np.sum(res.residual) / res.nfree
    rp_best, a_best, inc_best = res.params['rp'].value, res.params['a'].value, res.params['inc'].value
    t0_best = 0.
    print('Best fit parameters: rp={:.8f}; a={:.8f}; inc={:.8f}; t0={:.8f}'.format(rp_best, a_best, inc_best, t0_best))
   
    print('Minimization result: {}: {}; chi2={:.4f}'.format(res.success, res.message, chi2))
    
    #Produce a best fit model using the minimization results
    pm_best = bm.TransitParams()
    
    pm_best.t0 = 0.                #Time of transit centre
    
    pm_best.per = 1.               #Orbital period = 1 as phase folded
    pm_best.rp = rp_best           #Ratio of planet to stellar radius
    pm_best.a = a_best             #Semi-major axis (units of stellar radius)
    pm_best.inc = inc_best         #Orbital Inclination [deg]
    pm_best.ecc = 0.               #Orbital eccentricity (fix circular orbits)
    pm_best.w = 90.                #Longitude of periastron [deg] (unimportant as circular orbits)
    pm_best.u = [0.1, 0.3]         #Stellar LD coefficients
    pm_best.limb_dark="quadratic"  #LD model
    
    p_best = np.linspace(-0.2, 0.2, 10000)     #Produce a model LC using 
    m_best = bm.TransitModel(pm_best, p_best)  #the output best fit parameters
    f_best = m_best.light_curve(pm_best)
    
    if len(np.where(f_best < 1)[0]) > 0:
        p1 = p_best[np.where(f_best < 1)[0][0]]    #Phase of first contact
        p4 = p_best[np.where(f_best < 1)[0][-1]]   #Phase of final contact
        
        t_dur = (p4 - p1) * period *24             #Transit duration [hours]
        t_depth = (1 - f_best.min()) * 100         #Transit depth [percent]
        
    else:
		t_dur = np.nan
		t_depth = np.nan
		
    #Produce binned data set for plotting
    bw = 10 / (1440*period)                    #Bin width - 10 mins in units of phase
    p_bin, f_bin, e_bin = lc_bin(p_fit, f_fit, bw)
    
    #Produce plot of data and best fit model LC
    plt.figure(figsize=(9, 7.5))
    
    plt.plot(p_fit, f_fit, marker='o', color='gray', linestyle='none', markersize=3)
    plt.plot(p_bin, f_bin, 'ro', markersize=5)
    plt.plot(p_best, f_best, 'g--', linewidth=2)
    
    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.title('TIC  {}; T = {:.3f};  Rs = {:.3f} \n Depth: {:.5f}%;  Duration: {:5f} hours; Epoc; {:.6f} + {:.8f} \n (Rp/Rs): {:.6f};  chi2: {:.8f}; Period: {:.6f} days'.format(objid, Tmag, Rs, t_depth, t_dur, epoc, t0_best, rp_best, chi2, period))
    if np.isnan(t_dur) == False:
        plt.xlim((-3*p4, 3*p4))
        plt.ylim((f_bin.min()-t_depth/200, 1+t_depth/200))
    
    time1 = TIME()
    print("Time taken: {:.4f} s".format(time1-time0))

    plt.show()
    return p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t0_best, t_dur, t_depth, res
 
def lc_fit_spoc_lk(time, flux, flux_unflat, err, period, epoc, tdur, objid):
    '''
    Function to fit a batman model to an input lc datafile to find the best 
    fit system parameters
    '''

    time0 = TIME()
    
    phase = ((time - epoc)/period)%1  #Convert time values in to phases
    phase = np.array([p-1 if p > 0.8 else p for p in phase], dtype=float)
    
    p_fit = phase[phase < 0.2]  #Crop phase and flux arrays to only contain values
    f_fit = flux[phase < 0.2]   #in range (-0.2 ,  0.2)
    e_fit = err[phase < 0.2]
    f2_fit = flux_unflat[phase < 0.2]
    
    transit_indices = np.where(np.abs(p_fit * period) <= tdur / (2 * 24))	#Array indices of all phase/flux values during the transit
    FLUX_OOT = np.delete(f2_fit, transit_indices)				#"Out Of Transit" flux values

    median = np.median(FLUX_OOT)										#median of all out-of-transit flux values

    
    e_fit /= median
    
    params=Parameters()         #Parameter instance to hold fit parameters
    params.add('rp', value=0.05, min=0., max=1.)    #Planet:Star radius ratio
    params.add('a', value=10., min=0., max=100.)    #Semi-major axis
    params.add('inc', value=89., min=60., max=90.)  #Orbital inclination
    params.add('t0', value=0.0, min=-0.1, max=0.1) #Transit centre time
    
    res = minimize(lc_min, params, args=(p_fit, f_fit, e_fit), method='leastsq') #perform minimization
    chi2 = np.sum(res.residual) / res.nfree
    t0_best, rp_best, a_best, inc_best = res.params['t0'].value, res.params['rp'].value, res.params['a'].value, res.params['inc'].value

    print('Best fit parameters: rp={:.8f}; a={:.8f}; inc={:.8f}; t0={:.8f}'.format(rp_best, a_best, inc_best, t0_best))
   
    print('Minimization result: {}: {}; chi2={:.4f}'.format(res.success, res.message, chi2))
    
    #Produce a best fit model using the minimization results
    pm_best = bm.TransitParams()
    
    pm_best.t0 = 0.                #Time of transit centre
    
    pm_best.per = 1.               #Orbital period = 1 as phase folded
    pm_best.rp = rp_best           #Ratio of planet to stellar radius
    pm_best.a = a_best             #Semi-major axis (units of stellar radius)
    pm_best.inc = inc_best         #Orbital Inclination [deg]
    pm_best.ecc = 0.               #Orbital eccentricity (fix circular orbits)
    pm_best.w = 90.                #Longitude of periastron [deg] (unimportant as circular orbits)
    pm_best.u = [0.1, 0.3]         #Stellar LD coefficients
    pm_best.limb_dark="quadratic"  #LD model
    
    p_best = np.linspace(-0.2, 0.2, 10000)     #Produce a model LC using 
    m_best = bm.TransitModel(pm_best, p_best)  #the output best fit parameters
    f_best = m_best.light_curve(pm_best)
    
    if len(np.where(f_best < 1)[0]) > 0:
        p1 = p_best[np.where(f_best < 1)[0][0]]    #Phase of first contact
        p4 = p_best[np.where(f_best < 1)[0][-1]]   #Phase of final contact
        
        t_dur = (p4 - p1) * period *24             #Transit duration [hours]
        t_depth = (1 - f_best.min()) * 100         #Transit depth [percent]
        
    else:
		t_dur = np.nan
		t_depth = np.nan
		
    #Produce binned data set for plotting
    bw = 10 / (1440*period)                    #Bin width - 10 mins in units of phase
    p_bin, f_bin, e_bin = lc_bin(p_fit, f_fit, bw)
    
    #Produce plot of data and best fit model LC
    plt.figure(figsize=(9, 7.5))
    
    plt.plot(p_fit, f_fit, marker='o', color='gray', linestyle='none', markersize=1)
    plt.plot(p_bin, f_bin, 'ro', markersize=5)
    plt.plot(p_best, f_best, 'g--', linewidth=2)
    
    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.title('Depth: {:.5f}%;  Duration: {:5f} hours; Epoc; {:.6f} + {:.8f} \n (Rp/Rs): {:.6f};  chi2: {:.8f}; Period: {:.6f} days'.format(t_depth, t_dur, epoc, t0_best, rp_best, chi2, period))
    if np.isnan(t_dur) == False:
        plt.xlim((-3*p4, 3*p4))
        plt.ylim((f_bin.min()-t_depth/200, 1+t_depth/200))
    
    time1 = TIME()
    print("Time taken: {:.4f} s".format(time1-time0))

    plt.show()
    return p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t0_best, t_dur, t_depth, res
 
  
if __name__ == '__main__':
    
    args = ParseArgs()
    df = pandas.read_csv('/home/astro/phrvdf/tess_data_alerts/TOIs_20181016.csv', index_col='tic_id')		#.csv file containing info on parameters (period, epoch, ID, etc.) of all TOIs
    length=len(df.iloc[0])
    
    if args.pipe == 'spoc':
		
        time, flux, err, tic, Tmag, Rs = tess_LC_dataload_spoc(args.fn)
        
        epoc = df.loc[tic, 'Epoc']
        per = df.loc[tic, 'Period']
        tdur = df.loc[tic, 'Duration']
        toi = df.loc[tic, 'toi_id']
        
        print('Running fit for TIC {} (TOI-00{})'.format(tic, toi))
        
        print('Epoc: {}; Per: {}; tdur: {}'.format(epoc, per, tdur))
        
        
        if args.lightkurve == True:
            lc = TessLightCurve(time, flux)
            flat_lc = lc.flatten(window_length = args.wl)

            time_flat = flat_lc.time
            flux_flat = flat_lc.flux
            
            p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t0_best, t_dur, t_depth, res = lc_fit_spoc_lk(time, flux_flat, flux, err, per, epoc, tdur, tic)
    
        else: p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t0_best, t_dur, t_depth, res = lc_fit_spoc(time, flux, err, per, epoc, tdur, tic, Tmag, Rs)
    

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
from astropy.io.fits.card import Undefined
from astropy.constants import R_sun, R_jup, R_earth
from lightkurve import TessLightCurve
import pandas

def ParseArgs():
    '''
    Function to parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, nargs='*', help="Name of LC data file")
    parser.add_argument('-lk', '--lightkurve', action='store_true', help="Add this to use lightkurve to flatten the flux time series")
    parser.add_argument('-s', '--save', action='store_true', help="Add this to save the output plot instead of displaying")
    return parser.parse_args()
    
def lc_min(params, phase, flux, err):
	'''
	Function which calculates Chi2 value for a given set of input parameters.
	Function to be minimized to find best fit parameters
	'''
	
	#Define the system parameters for the batman LC model
	pm = bm.TransitParams()
	
	pm.t0 = params['t0'].value   #Time of transit centre
	pm.per = 1.                  #Orbital period = 1 as phase folded
	pm.rp = params['rp'].value   #Ratio of planet to stellar radius
	pm.a = params['a'].value             #Semi-major axis (units of stellar radius)
	pm.inc = params['inc'] .value          #Orbital Inclination [deg]
	pm.ecc = 0.                  #Orbital eccentricity (fix circular orbits)
	pm.w = 90.                   #Longitude of periastron [deg] (unimportant as circular orbits)
	pm.u = [0.3, 0.1]            #Stellar LD coefficients
	pm.limb_dark="quadratic"     #LD model
	
	#Initialize the batman LC model and compute model LC
	m = bm.TransitModel(pm, phase)
	f_model = m.light_curve(pm)
	residuals = (flux - f_model)**2/err**2
	return residuals

def lc_bin(time, flux, err, bin_width):
	'''
	Function to bin the data into bins of a given width. time and bin_width 
	must have the same units
	'''
	
	edges = np.arange(np.min(time), np.max(time), bin_width)
	dig = np.digitize(time, edges)
	time_binned = (edges[1:] + edges[:-1]) / 2
	flux_binned = np.array([np.nan if len(flux[dig == i]) == 0 else flux[dig == i].mean() for i in range(1, len(edges))])
	err_binned = np.array([np.nan if len(flux[dig == i]) == 0 else np.sqrt(np.sum(err[dig == i]**2))/len(err[dig == i]) for i in range(1, len(edges))])
	time_bin = time_binned[~np.isnan(err_binned)]
	err_bin = err_binned[~np.isnan(err_binned)]
	flux_bin = flux_binned[~np.isnan(err_binned)]	
	
	return time_bin, flux_bin, err_bin	

def tess_LC_dataload(file_name):
	'''Loads TESS LC data for a given object and uses quality flags etc. to remove bad points 
		'''
	#Load FITS file
	hdul = pyfits.open(file_name)               #Read in FITS file
	hdr = hdul[0].header                  		#Primary FITS header
	
	TIC = hdr['TICID']							#Loads TIC ID number from the fits header
	Tmag = hdr['TESSMAG']						#Loads TESS mag from fits header
	Rs = hdr['RADIUS']							#Loads stellar radius [solar radius] from fits header
	sector = hdr['SECTOR']						#Loads observing sector data was obtained in			
	
	if isinstance(Rs, Undefined):
		Rs = np.nan
	
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
	
	#Remove remaining nans
	time_nullremoved = time_nullremoved[~np.isnan(flux_nullremoved)]
	fluxerr_nullremoved = fluxerr_nullremoved[~np.isnan(flux_nullremoved)]
	flux_nullremoved = flux_nullremoved[~np.isnan(flux_nullremoved)]
	
	
	return time_nullremoved, flux_nullremoved, fluxerr_nullremoved, TIC, Tmag, Rs, sector
	
def lc_fit(time, flux, err, period, epoch, tdur, ticid, toiid, Tmag, Rs, sector):
    '''
    Function to fit a batman model to an input lc datafile to find the best 
    fit system parameters
    '''

    time0 = TIME()
    
    phase = ((time - epoch)/period)%1  #Convert time values in to phases
    phase = np.array([p-1 if p > 0.5 else p for p in phase], dtype=float)
    
    p_fit = phase[np.abs(phase) < 0.2]  #Crop phase and flux arrays to only contain values
    f_fit = flux[np.abs(phase) < 0.2]   #in range (-0.2 ,  0.2)
    e_fit = err[np.abs(phase) < 0.2]
    
    transit_indices = np.where(np.abs(p_fit * period) <= tdur / (2 * 24))	#Array indices of all phase/flux values during the transit
    FLUX_OOT = np.delete(f_fit, transit_indices)				#"Out Of Transit" flux values

    median = np.median(FLUX_OOT)										#median of all out-of-transit flux values

    f_fit /= median
    e_fit /= median
    
    params=Parameters()         #Parameter instance to hold fit parameters
    params.add('rp', value=0.05, min=0., max=1.)    #Planet:Star radius ratio
    params.add('a', value=10., min=0., max=100.)    #Semi-major axis
    params.add('inc', value=89., min=0., max=90.)  #Orbital inclination
    params.add('t0', value=0.0, min=-0.1, max=0.1) #Transit centre time
    
    res = minimize(lc_min, params, args=(p_fit, f_fit, e_fit), method='leastsq') #perform minimization
    chi2 = np.sum(res.residual) / res.nfree
    rp_best, a_best, inc_best, t0_best = res.params['rp'].value, res.params['a'].value, res.params['inc'].value, res.params['t0'].value

    print('Best fit parameters: rp={:.8f}; a={:.8f}; inc={:.8f}; t0={:.8f}'.format(rp_best, a_best, inc_best, t0_best))
   
    print('Minimization result: {}: {}; chi2={:.4f}'.format(res.success, res.message, chi2))
    
    #Produce a best fit model using the minimization results
    pm_best = bm.TransitParams()
    
    pm_best.t0 = t0_best           #Time of transit centre  
    pm_best.per = 1.               #Orbital period = 1 as phase folded
    pm_best.rp = rp_best           #Ratio of planet to stellar radius
    pm_best.a = a_best             #Semi-major axis (units of stellar radius)
    pm_best.inc = inc_best         #Orbital Inclination [deg]
    pm_best.ecc = 0.               #Orbital eccentricity (fix circular orbits)
    pm_best.w = 90.                #Longitude of periastron [deg] (unimportant as circular orbits)
    pm_best.u = [0.3, 0.1]         #Stellar LD coefficients
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
        print("Fitting failed to find a transit")
		
    #Produce binned data set for plotting
    bw = 10 / (1440*period)                    #Bin width - 10 mins in units of phase
    p_bin, f_bin, e_bin = lc_bin(p_fit, f_fit, e_fit, bw)
    
    #Produce plot of data and best fit model LC
    plt.figure(figsize=(9, 7.5))
    
    plt.plot(p_fit, f_fit, marker='o', color='gray', linestyle='none', markersize=3)
    plt.plot(p_bin, f_bin, 'ro', markersize=5)
    plt.plot(p_best, f_best, 'g--', linewidth=2)
    
    plt.xlabel('Phase')
    plt.ylabel('Flux')
    plt.title('TIC  {} (TOI-00{:.2f}); T = {:.3f}; Rs = {:.3f} \n Depth: {:.5f}%;  Duration: {:5f} hours; Epoch: {:.6f} + {:.8f} \n (Rp/Rs) = {:.4f}; Rp = {:.4f} Rjup;  chi2: {:.8f}; Period: {:.6f} days'.format(ticid, toiid, Tmag, Rs, t_depth, t_dur, epoch, t0_best, rp_best, rp_best * Rs * R_sun.value / R_jup.value, chi2, period))
    if np.isnan(t_dur) == False:
        plt.xlim((-3*p4, 3*p4))
        plt.ylim((f_bin.min()-t_depth/200, 1+t_depth/200))
    
    time1 = TIME()
    print("Time taken: {:.4f} s".format(time1-time0))

    if args.save:
        plt.savefig('/home/astro/phrvdf/tess_data_alerts/tess_LC_quickfit_plots/tess_{:.2f}_{}_lcfit_sector{}.png'.format(toiid, ticid, sector))
    
    else:
        plt.show()
    return p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t0_best, t_dur, t_depth, res
 
def lc_fit_lk(time, flux, flux_unflat, err, period, epoch, tdur, objid, Tmag, Rs):
    '''
    Function to fit a batman model to an input lc datafile to find the best 
    fit system parameters
    '''

    time0 = TIME()
    
    phase = ((time - epoch)/period)%1  #Convert time values in to phases
    phase = np.array([p-1 if p > 0.5 else p for p in phase], dtype=float)
    
    p_fit = phase[np.abs(phase) < 0.2]  #Crop phase and flux arrays to only contain values
    f_fit = flux[np.abs(phase) < 0.2]   #in range (-0.2 ,  0.2)
    e_fit = err[np.abs(phase) < 0.2]
    f2_fit = flux_unflat[phase < 0.2]
    
    transit_indices = np.where(np.abs(p_fit * period) <= tdur / (2 * 24))	#Array indices of all phase/flux values during the transit
    FLUX_OOT = np.delete(f2_fit, transit_indices)				#"Out Of Transit" flux values

    median = np.median(FLUX_OOT)										#median of all out-of-transit flux values

    
    e_fit /= median
    
    params=Parameters()         #Parameter instance to hold fit parameters
    params.add('rp', value=0.05, min=0., max=1.)    #Planet:Star radius ratio
    params.add('a', value=10., min=0., max=100.)    #Semi-major axis
    params.add('inc', value=89., min=0., max=90.)  #Orbital inclination
    params.add('t0', value=0.0, min=-0.1, max=0.1) #Transit centre time
    
    res = minimize(lc_min, params, args=(p_fit, f_fit, e_fit), method='leastsq') #perform minimization
    chi2 = np.sum(res.residual) / res.nfree
    t0_best, rp_best, a_best, inc_best = res.params['t0'].value, res.params['rp'].value, res.params['a'].value, res.params['inc'].value

    print('Best fit parameters: rp={:.8f}; a={:.8f}; inc={:.8f}; t0={:.8f}'.format(rp_best, a_best, inc_best, t0_best))
   
    print('Minimization result: {}: {}; chi2={:.4f}'.format(res.success, res.message, chi2))
    
    #Produce a best fit model using the minimization results
    pm_best = bm.TransitParams()
    
    pm_best.t0 = t0_best           #Time of transit centre
    
    pm_best.per = 1.               #Orbital period = 1 as phase folded
    pm_best.rp = rp_best           #Ratio of planet to stellar radius
    pm_best.a = a_best             #Semi-major axis (units of stellar radius)
    pm_best.inc = inc_best         #Orbital Inclination [deg]
    pm_best.ecc = 0.               #Orbital eccentricity (fix circular orbits)
    pm_best.w = 90.                #Longitude of periastron [deg] (unimportant as circular orbits)
    pm_best.u = [0.3, 0.1]         #Stellar LD coefficients
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
    plt.title('TIC  {}; T = {:.3f}; Rs = {:.3f} \n Depth: {:.5f}%;  Duration: {:5f} hours; Epoch: {:.6f} + {:.8f} \n Rp = {:.4f} Rsun = {:.4f} Rjup;  chi2: {:.8f}; Period: {:.6f} days'.format(objid, Tmag, Rs, t_depth, t_dur, epoch, t0_best, rp_best, rp_best * Rs * R_sun.value / R_jup.value, chi2, period))
    if np.isnan(t_dur) == False:
        plt.xlim((-3*p4, 3*p4))
        plt.ylim((f_bin.min()-t_depth/200, 1+t_depth/200))
    
    time1 = TIME()
    print("Time taken: {:.4f} s".format(time1-time0))

    plt.show()
    return p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t0_best, t_dur, t_depth, res
 
def lc_detrend(time, flux, err, per, epoch, tdur):
    '''
    Function to flatten LCs. Removes transits and then fits a moving 
    average trend using a Savitzky-Golay filter - implemented using lightkurve
    '''
    
    flux_trend = np.zeros_like(flux) + flux
    phase_trend = (time - epoch) / per
    
    n_trans = np.int((time[-1] - epoch) / per + 1)
    for i in range(n_trans):
        trans = np.array((time[np.abs(phase_trend - i) < tdur/24/per], flux[np.abs(phase_trend - i) < tdur/24/per]))
        length = np.int(len(trans[0])/4)
		
        m = (np.mean(trans[1, -1*length:]) - np.mean(trans[1, :length])) / (np.mean(trans[0, -1*length:]) - np.mean(trans[0, :length]))
        c = np.mean(trans[1, :length]) - m*np.mean(trans[0, :length])
		
        flux_trend[np.where(np.abs(phase_trend - i) < tdur/24/per)] = m*trans[0] + c
	
    flat_lc, trend_lc = TessLightCurve(time, flux_trend).flatten(window_length=101, return_trend=True)
    
    flux_flat = flux/trend_lc.flux
    err_flat = err/trend_lc.flux
    return flux_flat, err_flat, trend_lc.flux, flux_trend

if __name__ == '__main__':
    
    args = ParseArgs()
    df = pandas.read_csv('/home/astro/phrvdf/tess_data_alerts/toi_ephems.csv', index_col='tic_id')		#.csv file containing info on parameters (period, epoch, ID, etc.) of all TOIs
    length=len(df.iloc[0])
    
    for fn in args.fn:
    
        time, flux, err, tic, Tmag, Rs, sector = tess_LC_dataload(fn)
    
        df2 = df.loc[tic]
        if len(df2) == length:
    
            epoch = df.loc[tic, 'Epoc']
            per = df.loc[tic, 'Period']
            tdur = df.loc[tic, 'Duration']
            toi = df.loc[tic, 'toi_id']
        
            print('Running fit for TIC {} (TOI-00{})'.format(tic, toi))
            print('Epoch: {}; Per: {}; tdur: {}'.format(epoch, per, tdur))
        
            if args.lightkurve: flux, err, trend_flux, flux_trend = lc_detrend(time, flux, err, per, epoch, tdur)
       
            p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t0_best, t_dur, t_depth, res = lc_fit(time, flux, err, per, epoch, tdur, tic, toi, Tmag, Rs, sector)
            
            df.loc[tic, 't0_best'] = t0_best
            df.loc[tic, 'rp_best'] = rp_best
            df.loc[tic, 'Rp [Rjup]'] = rp_best * Rs * R_sun.value / R_jup.value
            df.loc[tic, 'Rp [Rearth]'] = rp_best * Rs * R_sun.value / R_earth.value
            df.loc[tic, 'Rs [Rsun]'] = Rs
            df.loc[tic, 'a_best'] = a_best
            df.loc[tic, 'inc_best'] = inc_best
            df.loc[tic, 'b_best'] = a_best * np.cos(inc_best * np.pi / 180.)
            df.loc[tic, 't_dur'] = t_dur
            df.loc[tic, 't_depth'] = t_depth * 10000
            
            
        else:
            for j in range(len(df2)):
            
                df3 = df2.iloc[j]
            
                epoch = df3.loc['Epoc']
                per = df3.loc['Period']
                tdur = df3.loc['Duration']
                toi = df3.loc['toi_id']
        
                print('Running fit for TIC {} (TOI-00{})'.format(tic, toi))
                print('Epoch: {}; Per: {}; tdur: {}'.format(epoch, per, tdur))
            
                if args.lightkurve: flux, err, trend_flux, flux_trend = lc_detrend(time, flux, err, per, epoch, tdur)
                p_best, f_best, p_fit, f_fit, e_fit, rp_best, a_best, inc_best, t0_best, t_dur, t_depth, res = lc_fit(time, flux, err, per, epoch, tdur, tic, toi, Tmag, Rs, sector)
                
                df.loc[df['toi_id'] == toi, 't0_best'] = t0_best
                df.loc[df['toi_id'] == toi, 'rp_best'] = rp_best
                df.loc[df['toi_id'] == toi, 'Rp [Rjup]'] = rp_best * Rs * R_sun.value / R_jup.value
                df.loc[df['toi_id'] == toi, 'Rp [Rearth]'] = rp_best * Rs * R_sun.value / R_earth.value
                df.loc[df['toi_id'] == toi, 'Rs [Rsun]'] = Rs
                df.loc[df['toi_id'] == toi, 'a_best'] = a_best
                df.loc[df['toi_id'] == toi, 'inc_best'] = inc_best
                df.loc[df['toi_id'] == toi, 'b_best'] = a_best * np.cos(inc_best * np.pi / 180.)
                df.loc[df['toi_id'] == toi, 't_dur'] = t_dur
                df.loc[df['toi_id'] == toi, 't_depth'] = t_depth * 10000
    df.to_csv('/home/astro/phrvdf/tess_data_alerts/toi_ephems.csv')        
            
            
            
            
            
            
            
            
            
            
            
            
            
            

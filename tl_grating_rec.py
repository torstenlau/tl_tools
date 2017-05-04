# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:58:09 2015

@author: torstenl
"""

from PIL import Image
import numpy as np
import pyqtgraph as pg

pg.QtGui.QApplication.closeAllWindows()


def phase_scane_four_par_sine_fit(exp_stack,period_guess=0):#period_guess is the guessed number of steps required to cover one phase period. If not given or if zero, exp_stack.shape[2] is used
    #Applies an iterative sine fit to a stack of phase scan images
    if period_guess==0:
        period_guess=exp_stack.shape[2]
    Nsteps=exp_stack.shape[2]
    omega=2*np.pi/period_guess
    fft_stack=np.fft.fft(exp_stack, n=period_guess, axis=2)
    A=2*np.real(fft_stack[:,:,1])/np.minimum(period_guess,fft_stack.shape[2])
    B=-2*np.imag(fft_stack[:,:,1])/np.minimum(period_guess,fft_stack.shape[2])
    C=np.real(fft_stack[:,:,0]/np.minimum(period_guess,fft_stack.shape[2]))
    
    x=np.zeros((exp_stack.shape[0],exp_stack.shape[1],4))
    x[:,:,0]=A
    x[:,:,1]=B
    x[:,:,2]=C

    fit_ok=0
    delta_omega=0
    while fit_ok!=1:
        t=np.arange(Nsteps)
        omega=omega+delta_omega
        col1=np.cos(omega*t)
        col2=np.sin(omega*t)
        col3=t*np.sin(omega*t)
        col4=t*np.cos(omega*t)
        for coord1 in np.arange(exp_stack.shape[0]):
            for coord2 in np.arange(exp_stack.shape[1]):
                D_tr=np.array([col1,col2,np.ones((Nsteps)),-x[coord1,coord2,0]*col3+x[coord1,coord2,1]*col4])
                D=np.transpose(D_tr)
                if not np.linalg.det(np.dot(D_tr,D))<1e-10:
                    x[coord1,coord2,:]=np.dot(np.dot(np.linalg.inv(np.dot(D_tr,D)),D_tr),np.reshape(exp_stack[coord1,coord2,:],(Nsteps,)))
                else:
                    x[coord1,coord2,3]=0
        delta_omega=np.median(x[:,:,3])
        pos_frac=float(np.sum(x[:,:,3]>0))/float(x[:,:,3].size)
        if pos_frac>0.49 and pos_frac<0.51:
            fit_ok=1
        else:
            fit_ok=0
    C=x[:,:,2]
    vis=np.sqrt(x[:,:,0]**2+x[:,:,1]**2)/C
    phase=np.angle(x[:,:,0]+1j*x[:,:,1])
    return C,vis,phase,omega

def grating_projs_from_img_format(loaddir,sample_prefixes,img_nos,flat_prefix,flat_nos,no_of_digits,suffix,period_guess):
    
    im = np.array(Image.open(loaddir+flat_prefix+"%04d" % img_nos[0]+suffix))
    exp_stack=np.zeros(im.shape+(img_nos.size, ))
    exp_stack[:,:,0]=im

    for load_iter_no in np.arange(1,flat_nos.size):
        im = Image.open(loaddir+flat_prefix+"%04d" % flat_nos[load_iter_no]+suffix)
        exp_stack[:,:,load_iter_no] = np.array(im)
    C_f,vis_f,phase_f,omega_f=phase_scane_four_par_sine_fit(exp_stack,period_guess)

    T_array=np.zeros(C_f.shape + (len(sample_prefixes), ))
    df_array=np.zeros(C_f.shape + (len(sample_prefixes), ))
    d_phi_array=np.zeros(C_f.shape + (len(sample_prefixes), ))
    for sample_no in np.arange(len(sample_prefixes)):
        for load_iter_no in np.arange(img_nos.size):
            im = Image.open(loaddir+sample_prefixes[sample_no]+"%04d" % img_nos[load_iter_no]+suffix)
            exp_stack[:,:,load_iter_no] = np.array(im)
        C_s,vis_s,phase_s,omega_s=phase_scane_four_par_sine_fit(exp_stack,period_guess)
        T_array[:,:,sample_no]=C_s/C_f
        df_array[:,:,sample_no]=vis_s/vis_f
        phase_shift=np.mod(phase_s-phase_f+np.pi,2*np.pi)-np.pi
        mean_phase_angle=np.angle(np.mean(np.exp(1j*phase_shift)))
        d_phi_array[:,:,sample_no]=np.mod(phase_shift-mean_phase_angle+np.pi,2*np.pi)-np.pi
    
    return T_array, df_array, d_phi_array


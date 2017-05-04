# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:13:01 2016

@author: torstenl
"""

import numpy as np
import scipy.interpolate
import scipy.optimize
import pyqtgraph as pg
import tl_ct_rec as tlct
import tl_tools as tl

def optimize_geodesic_line(img,fixed_point):
    #img is assumed to be a complex gradient image
    img_interp=scipy.interpolate.RectBivariateSpline(np.arange(img.shape[0]), np.arange(img.shape[1]), img)
    def img_interp_extrapol(length_coord,line_heights):
        out_line=np.zeros(line_heights.shape,dtype='complex')
        out_line[line_heights>img.shape[0]]=1j*np.mean(np.abs(img))
        out_line[line_heights<0]=1j*np.mean(np.abs(img))
        out_line[(0>line_heights)*(line_heights>img.shape[0])]=img_interp(length_coord[(0>line_heights)*(line_heights>img.shape[0])],line_heights[(0>line_heights)*(line_heights>img.shape[0])])
        return out_line
    line_heights=np.ones((img.shape[1],))*fixed_point[1]
    def curve_punish(line_heights):
        print(img.shape)
        print(line_heights.shape)
        
        img_gradients=img_interp_extrapol(np.arange(img.shape[1]),line_heights)
        line_gradients=np.gradient(line_heights)
        complex_line_orientations=(1+1j*line_gradients)/np.sqrt(1+line_gradients**2)
        gradient_dot=img_gradients/complex_line_orientations
        img_punish=np.abs(np.real(gradient_dot))
        print(np.sum(img_punish))
        return img_punish
    line_heights=scipy.optimize.minimize(curve_punish, line_heights)
    punishment=curve_punish(line_heights)
    print(punishment)
    return line_heights

def geodesic_propagation(grad_img,start_height,m):
    real_img_interp=scipy.interpolate.RectBivariateSpline(np.arange(grad_img.shape[0]), np.arange(grad_img.shape[1]), np.real(grad_img))
    imag_img_interp=scipy.interpolate.RectBivariateSpline(np.arange(grad_img.shape[0]), np.arange(grad_img.shape[1]), np.imag(grad_img))
    height_vector=np.zeros(grad_img.shape[0],dtype=float)
    grad_vector=np.zeros(grad_img.shape[0],dtype=float)
    height_vector[0]=start_height
    grad_vector[0]=0
    grad_img[0:10,295:305]=80
    tlct.plot_df_slice(grad_img)
    for step_no in np.arange(1,grad_img.shape[0]):
        real_field_gradient=real_img_interp(step_no-1,height_vector[step_no-1])
        imag_field_gradient=imag_img_interp(step_no-1,height_vector[step_no-1])
        field_gradient=real_field_gradient
        print(field_gradient)
        prev_step_force=field_gradient-grad_vector[step_no-1]
        grad_vector[step_no]=grad_vector[step_no-1]+prev_step_force/m
        height_vector[step_no]=height_vector[step_no-1]+grad_vector[step_no]
    return height_vector
    
def plot_image_with_curve(img,curve):
    
    pg.mkQApp()
    print(img.shape)
    win = pg.GraphicsLayoutWidget()
    win.setWindowTitle('pyqtgraph example: Image Analysis')
    p1 = win.addPlot()
    # Item for displaying image data
    img_item = pg.ImageItem()
    p1.addItem(img_item)
    #curve_plot=pg.PlotDataItem((0,1,2,3,4,5),(3,6,1,5,8,9))
    curve_plot=pg.PlotDataItem(np.arange(img.shape[0]),curve)
    p1.addItem(curve_plot)
    win.show()
    img_item.setImage(img)
    print(img.shape)
    stop
    #pg.image(img,title='curve_img')
    '''    
    rgb_img=tl.tile(img,(1,1,3))#np.zeros(img.shape+(3,),dtype=float)
    int_curve=np.round(curve).astype('int16')
    rgb_img[np.arange(img.shape[0]),int_curve,0]=1
    pg.image(rgb_img)
    '''

def fiber_curve_fitter(img,fix_points,no_of_fit_points,no_of_eval_points):
    global img_interp
    global eval_point_positions
    global fit_point_positions
    #global fit_points
    global fixed_points
    fixed_points=fix_points

    img_interp=scipy.interpolate.RectBivariateSpline(np.arange(img.shape[0]), np.arange(img.shape[1]), img)
    #img_interp=scipy.interpolate.interp2d(np.arange(img.shape[0]), np.arange(img.shape[1]), img)
    #fix_point_interp_func=scipy.interpolate.UnivariateSpline(fix_points[:,0], fix_points[:,1])
    fix_point_interp_func=scipy.interpolate.interp1d(fix_points[:,0], fix_points[:,1], kind='cubic')

    fit_point_positions=np.linspace(0,img.shape[0]-1,num=no_of_fit_points)
    fit_points=fix_point_interp_func(fit_point_positions)
    
    eval_point_positions=np.linspace(0,img.shape[0]-1,num=no_of_eval_points)

    fit_eval=fit_evaluator(fit_points)
    print(fit_eval)

    optim_result=scipy.optimize.minimize(fit_evaluator, fit_points,options={'maxiter': 30} )
    fit_points=optim_result.x
    fit_eval=fit_evaluator(fit_points)
    print(fit_eval)
    return fit_point_positions, fit_points

def fit_evaluator(fit_points):
#    fit_point_interp_func=scipy.interpolate.interp1d(fit_point_positions, fit_points, kind='cubic', fill_value='extrapolate')
    fit_point_interp_func=scipy.interpolate.interp1d(fit_point_positions, fit_points, kind='cubic')

    # First look at the image values along the curve
    eval_points=fit_point_interp_func(eval_point_positions)
    img_val_eval_curve=img_interp.ev(eval_point_positions,eval_points)
    #pg.plot(np.gradient(img_val_eval_curve)**2)
    #img_val_curve_gradient2=np.sum(np.gradient(img_val_eval_curve)**2)/1e4
    img_val_curve_gradient2=np.sum((772-img_val_eval_curve)**2)*1e-6
    
    # Then look at curvature of curve
    curvature=np.sum(np.abs(np.gradient(np.gradient(eval_points))))*1e-3
    #curvature=0
    # Finally, look at closeness to fixed points

    fix_point_dev=np.sum((fixed_points[:,1]-fit_point_interp_func(fixed_points[:,0]))**2)*1e-2
    
    del(fit_point_interp_func)
    print((img_val_curve_gradient2,curvature,fix_point_dev))
    
    #print(img_val_curve_gradient2)
    #print(curvature)
    #print(fix_point_dev)
    
    return(img_val_curve_gradient2+curvature+fix_point_dev)
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:20:06 2015

Axes are defined from the detector plane orientation. All functions in this script use the following axis definitions:
first axis: downwards, lies in the detector plane.
second axis: horizontal, lies in the detector plane. Goes from left to right, when seen from the detector towards the source.
third axis: horizontal, perpendicular to the detector. Goes from the side of the detector plane where the source is placed, towards the other side of the detector
sample rotation axis: The first axis, positive rotation is clockwise, when seen from above, consistent with the downward orientation of the first axis.

source_dist: third axis-projected distance from source to rotation center
det_dist: third axis-projected distance from detector to rotation center

source_pos: detector pixel coordinates of source when projected down on the detector plane (projecting along the third axis)
rot_center: horizontal detector pixel coordinate of rotation center when projected down on the detector (projecting plane along the third axis)

By definition, the rotation center is in the middle of the horizontal plane of the tomogram
@author: torstenl
"""
import numpy as np
import sys
sys.path.append('/home/torstenl/Documents/python/tl_modules')
import tl_tools as tl
import pyqtgraph as pg

def fdk_rec(projstack, angles, source_dist, source_pos, rot_pos, det_dist, pixel_size):
    tomo=np.zeros((projstack.shape[0],projstack.shape[1],projstack.shape[1]),dtype='float')
    for proj_iter_no in np.arange(len(angles)):
        tomo+=back_proj(projstack[:,:,proj_iter_no], np.arange(projstack.shape[0]), angles[proj_iter_no], source_dist, source_pos, rot_pos, det_dist, pixel_size, proj_filter='fdk')*np.pi/len(angles)
        print(proj_iter_no)
        '''
        print(proj_iter_no)
        print(np.sum(dfstack[:,:,proj_iter_no]))
        pg.image(tomo[2,:,:])
        pg.image(dfstack[:,:,proj_iter_no])
        stop
        '''
    return tomo

def aniso_rec(dfstack, angles, no_of_ft_comps, source_dist, source_pos, rot_pos, det_dist, pixel_size, tomo_mask='none', k_punishment=0, fdk_start=0):
    tomo=np.zeros((dfstack.shape[0],dfstack.shape[1],dfstack.shape[1],no_of_ft_comps),dtype='complex64')
    if fdk_start==1:
        for proj_iter_no in np.arange(len(angles)):
            tomo[:,:,:,0]+=back_proj(dfstack[:,:,proj_iter_no], np.arange(dfstack.shape[0]), angles[proj_iter_no], source_dist, source_pos, rot_pos, det_dist, pixel_size, proj_filter='none')*2*np.pi/len(angles)
            if np.mod(proj_iter_no,np.round(angles.size/10)) == 0:
                print('fdk backprojecting angle no ' + str(proj_iter_no))
        tomo=tomo/2
        old_tomo_sum=np.sum(tomo)
        tomo[tomo<0]=0
        tomo=tomo*old_tomo_sum/np.sum(tomo)
    
    dtomo_scale, std_vector=dtomo_scaler(np.zeros(tomo.shape), tomo, dfstack, angles, source_dist, source_pos, rot_pos, det_dist, pixel_size)
    print('tot_std')
    print(np.sqrt(np.mean(std_vector**2)))

    for iter_no in range(15):
        dSE_dtomo_map=aniso_error_backprojector(tomo, dfstack, angles, source_dist, source_pos, rot_pos, det_dist, pixel_size, mask=tomo_mask)

        tomo=k_punisher(tomo,k_punishment)
        dSE_dtomo_map=k_punisher(dSE_dtomo_map,k_punishment)
        
        dtomo_scale, std_vector=dtomo_scaler(dSE_dtomo_map, tomo, dfstack, angles, source_dist, source_pos, rot_pos, det_dist, pixel_size)

        print(dtomo_scale)
        print('tot_std')
        print(np.sqrt(np.mean(std_vector**2)))
 
        tomo+=dSE_dtomo_map*dtomo_scale

        tomo=voxel_similarizer(tomo)*tl.tile(tomo_mask,(1,1,1,tomo.shape[3]))

    plot_df_slice(tomo[0,:,:,1],colormap='rg',img_title='tomo1')
    pg.image(np.real(tomo[0,:,:,0]),title='tomo0')
    return tomo

def aniso_error_backprojector(tomo, dfstack, angles, source_dist, source_pos, rot_pos, det_dist, pixel_size, mask='radial'):
    # assumes angles to be equidistantly distributed on the interval [0 ; 2pi]
    if mask=='radial':
        mask=radial_mask_tomo(tomo.shape)
    bck_proj_tomo=np.zeros(tomo.shape,dtype='complex64')
    
    for angle_no in np.arange(angles.size):
        prop_angle_slice, prop_angle_step_slice=prop_angles(tomo.shape, angles[angle_no], (angles[-1]-angles[0])/(np.size(angles)-1), source_dist, source_pos, rot_pos, det_dist, pixel_size)

        current_proj, map_coeff_sum=forward_proj(tomo, np.arange(tomo.shape[0]), angles[angle_no], source_dist, source_pos, rot_pos, det_dist, pixel_size, mask=mask)
        normalised_res_proj=(dfstack[:,:,angle_no]-current_proj)/map_coeff_sum
        normalised_res_proj[map_coeff_sum==0]=0
        spatial_error_bck_proj=back_proj(normalised_res_proj, np.arange(tomo.shape[0]), angles[angle_no], source_dist, source_pos, rot_pos, det_dist, pixel_size)*mask
        if np.mod(angle_no,np.round(angles.size/10)) == 0:
            print('error backprojecting angle no ' + str(angle_no))
        bck_proj_tomo+=ft_comp_proj(spatial_error_bck_proj,tomo.shape[3],prop_angle_slice)*np.tile(np.reshape(np.abs(prop_angle_step_slice),(tomo.shape[1:3]+(1,))),(tomo.shape[0],1,1,tomo.shape[3]))/(2*np.pi)
        
    return bck_proj_tomo

def forward_projs(tomo, slice_nos, angles, source_dist, source_pos, rot_pos, det_dist, pixel_size,mask='full'):
    proj_stack=np.zeros((len(slice_nos),np.size(tomo,axis=1),len(angles)),dtype='float32')
    for angle_no in range(len(angles)):
        print angle_no
        proj_stack[:,:,angle_no]=forward_proj(tomo, slice_nos, angles[angle_no], source_dist, source_pos, rot_pos, det_dist, pixel_size, mask='full', return_map_coeff_sum=False)
    return proj_stack

def forward_proj(tomo, slice_nos, angle, source_dist, source_pos, rot_pos, det_dist, pixel_size, mask='full', return_map_coeff_sum=True):
    proj=np.zeros((len(slice_nos),np.size(tomo,axis=1)),dtype='float32')
    if return_map_coeff_sum:
        map_coeff_sum=np.zeros((len(slice_nos),np.size(tomo,axis=1)),dtype='float32')

    pixel_coord2_slice, pixel_span_slice, prop_angle_slice, mag_slice=hor_plane_voxel_coordinates(tomo.shape, angle, source_dist, source_pos, rot_pos, det_dist, pixel_size)
    voxel_size=pixel_size*source_dist/(source_dist+det_dist)
    
    # Calculating current tomogram
    if len(tomo.shape)==4:
        tomo=ft_comp_collapser(tomo,prop_angle_slice)
    elif len(tomo.shape)!=3:
        print('tomogram dimensionality neither 3 nor 4')
        stop

    # making disk-like mask slice if necessary
    if mask=='radial':
        mask=radial_mask_tomo(tomo.shape)
    elif mask=='full':
        mask=np.ones(tomo.shape[0:3],dtype='float32')

    for slice_no in np.arange(len(slice_nos)):
        pixel_coord1_slice=((slice_no-source_pos[0])*voxel_size)*mag_slice/pixel_size+source_pos[0]
        map_coord1_slices, map_coord2_slices, geometrical_weighting_slices = geometrical_voxel_weighting(pixel_coord1_slice,pixel_coord2_slice,pixel_span_slice)
        proj+=proj_mapper(map_coord1_slices, map_coord2_slices, geometrical_weighting_slices, tomo[slice_no,:,:], proj.shape)
        if return_map_coeff_sum:
            map_coeff_sum+=proj_mapper(map_coord1_slices, map_coord2_slices, geometrical_weighting_slices, mask[slice_no,:,:], proj.shape)
    # multiplying by propagation distance through the individual voxel
    proj=proj_filters(proj,source_dist,voxel_size,source_pos,'cone_fwd_post_weight')*voxel_size
    if return_map_coeff_sum:
        map_coeff_sum=proj_filters(map_coeff_sum,source_dist,voxel_size,source_pos,'cone_fwd_post_weight')*voxel_size
        return proj, map_coeff_sum
    else:    
        return proj
    
def back_proj(proj, slice_nos, angle, source_dist, source_pos, rot_pos, det_dist, pixel_size, proj_filter='none'):
    voxel_size=pixel_size*source_dist/(source_dist+det_dist)
    tomo=np.zeros((slice_nos.size, proj.shape[1], proj.shape[1]),dtype='float32')
    if proj_filter=='fdk':
        #old_proj=proj
        #pg.image(old_proj,title='old proj')
        proj=proj_filters(proj,source_dist,voxel_size,source_pos,'fdk_prefilter')
        #pg.image(proj)
        #stop
    elif proj_filter=='none':
        proj=proj_filters(proj,source_dist,voxel_size,source_pos,'cone_bck_pre_weight')
    
    pixel_coord2_slice, pixel_span_slice, prop_angle_slice, mag_slice=hor_plane_voxel_coordinates(tomo.shape, angle, source_dist, source_pos, rot_pos, det_dist, pixel_size)
    padded_proj=np.pad(proj,(1,),'constant', constant_values=(0,)) #used to map zeros for voxel projected outside the detector screen

    for slice_iter_no in np.arange(slice_nos.size):
        slice_no=slice_nos[slice_iter_no]
        pixel_coord1_slice=((slice_no-source_pos[0])*voxel_size)*mag_slice/pixel_size+source_pos[0]
        map_coord1_slices, map_coord2_slices, geometrical_weighting_slices = geometrical_voxel_weighting(pixel_coord1_slice,pixel_coord2_slice,pixel_span_slice)

        padded_map_coord1_slices=np.clip(map_coord1_slices, -1, proj.shape[0])+1
        padded_map_coord2_slices=np.clip(map_coord2_slices, -1, proj.shape[1])+1
        
        tomo[slice_iter_no,:,:]=np.sum(padded_proj[padded_map_coord1_slices,padded_map_coord2_slices]*geometrical_weighting_slices,axis=(2,3))
    #pg.image(tomo[2,:,:])
    #stop
    return tomo

def ft_comp_collapser(tomo,prop_angle_slice):
    # input tomo assumed to be 4D
    prop_angle_slice=np.reshape(prop_angle_slice,(1,prop_angle_slice.shape[0],prop_angle_slice.shape[1],1))
    ft_order_vector=2*np.arange(tomo.shape[3]) #DF signal is assumed pi-periodic, so only even ft-components are considered.
    ft_exponential=1j*np.tile(ft_order_vector,(np.size(tomo,axis=0),np.size(tomo,axis=1),np.size(tomo,axis=1),1))*np.tile(prop_angle_slice,(tomo.shape[0],1,1,tomo.shape[3]))
    tomo=tomo*np.exp(ft_exponential)
    tomo[:,:,:,0]=tomo[:,:,:,0]/2
    tomo=2*np.real(np.sum(tomo,axis=3))
    #tomo=np.real(np.sum(tomo,axis=3))
    return tomo

def ft_comp_proj(tomo,no_of_ft_comps,prop_angle_slice):
    # input tomo assumed to be 3D
    prop_angle_slice=np.reshape(prop_angle_slice,(1,prop_angle_slice.shape[0],prop_angle_slice.shape[1],1))
    tomo=tl.tile(tomo,(1,1,1,no_of_ft_comps))
    ft_order_vector=2*np.arange(tomo.shape[3]) #DF signal is assumed pi-periodic, so only even ft-components are considered.
    ft_exponential=1j*np.tile(ft_order_vector,(np.size(tomo,axis=0),np.size(tomo,axis=1),np.size(tomo,axis=1),1))*np.tile(prop_angle_slice,(tomo.shape[0],1,1,tomo.shape[3]))
    tomo=tomo*np.exp(-ft_exponential)
    return tomo
    
def hor_plane_voxel_coordinates(tomo_shape, angle, source_dist, source_pos, rot_pos, det_dist, pixel_size):
    voxel_size=pixel_size*source_dist/(source_dist+det_dist)
    center_ray_rot_center_dist=(rot_pos-source_pos[1])*voxel_size

    # horizontal plane coordinates
    coord2_slice, coord3_slice = np.mgrid[0:tomo_shape[1],0:tomo_shape[1]].astype('float32')

    coord2_slice=(coord2_slice-(float(tomo_shape[1])-1)/2)*voxel_size
    coord3_slice=(coord3_slice-(float(tomo_shape[1])-1)/2)*voxel_size
    setup_coord2_slice=-np.sin(angle)*coord3_slice+np.cos(angle)*coord2_slice #displacement from rotation center along the second axis of the setup coordinate system (parallel to the detector screen, in the horizontal plane)
    setup_coord3_slice=np.cos(angle)*coord3_slice+np.sin(angle)*coord2_slice #displacement from rotation center along the third axis of the setup coordinate system (perpendicular to the detector screen)

    mag_slice=(det_dist+source_dist)/(source_dist+setup_coord3_slice)
    pixel_coord2_slice=(setup_coord2_slice+(rot_pos-source_pos[1])*voxel_size)*mag_slice/pixel_size+source_pos[1]
    
    pixel_span_slice=source_dist/(source_dist+setup_coord3_slice)
    # Calculating current tomogram
    fan_angle_slice=np.arcsin(-(setup_coord2_slice+center_ray_rot_center_dist)/(source_dist+setup_coord3_slice))
    prop_angle_slice=fan_angle_slice-angle
    return pixel_coord2_slice, pixel_span_slice, prop_angle_slice, mag_slice
    
def prop_angles(tomo_shape, angle, angle_step, source_dist, source_pos, rot_pos, det_dist, pixel_size):
    voxel_size=pixel_size*source_dist/(source_dist+det_dist)
    center_ray_rot_center_dist=(rot_pos-source_pos[1])*voxel_size

    # horizontal plane coordinates
    coord2_slice, coord3_slice = np.mgrid[0:tomo_shape[1],0:tomo_shape[1]].astype('float32')
    coord2_slice=(coord2_slice-(float(tomo_shape[1])-1)/2)*voxel_size
    coord3_slice=(coord3_slice-(float(tomo_shape[1])-1)/2)*voxel_size

    # Calculating current angles
    setup_coord2_slice=-np.sin(angle)*coord3_slice+np.cos(angle)*coord2_slice #displacement from rotation center along the second axis of the setup coordinate system (parallel to the detector screen, in the horizontal plane)
    setup_coord3_slice=np.cos(angle)*coord3_slice+np.sin(angle)*coord2_slice #displacement from rotation center along the third axis of the setup coordinate system (perpendicular to the detector screen)
    fan_angle_slice=np.arcsin(-(setup_coord2_slice+center_ray_rot_center_dist)/(source_dist+setup_coord3_slice))
    prop_angle_slice=fan_angle_slice-angle
    
    # Calculating angle_steps
    setup_coord2_slice=-np.sin(angle-angle_step/2)*coord3_slice+np.cos(angle-angle_step/2)*coord2_slice #displacement from rotation center along the second axis of the setup coordinate system (parallel to the detector screen, in the horizontal plane)
    setup_coord3_slice=np.cos(angle-angle_step/2)*coord3_slice+np.sin(angle-angle_step/2)*coord2_slice #displacement from rotation center along the third axis of the setup coordinate system (perpendicular to the detector screen)
    fan_angle_slice=np.arcsin(-(setup_coord2_slice+center_ray_rot_center_dist)/(source_dist+setup_coord3_slice))
    start_prop_angle=fan_angle_slice-(angle-angle_step/2)
    setup_coord2_slice=-np.sin(angle+angle_step/2)*coord3_slice+np.cos(angle+angle_step/2)*coord2_slice #displacement from rotation center along the second axis of the setup coordinate system (parallel to the detector screen, in the horizontal plane)
    setup_coord3_slice=np.cos(angle+angle_step/2)*coord3_slice+np.sin(angle+angle_step/2)*coord2_slice #displacement from rotation center along the third axis of the setup coordinate system (perpendicular to the detector screen)
    fan_angle_slice=np.arcsin(-(setup_coord2_slice+center_ray_rot_center_dist)/(source_dist+setup_coord3_slice))
    prop_angle_step_slice=fan_angle_slice-(angle+angle_step/2)-start_prop_angle

    return prop_angle_slice, prop_angle_step_slice
    
def geometrical_voxel_weighting(pixel_coord1_slice,pixel_coord2_slice,pixel_span_slice):
    # returns coordinate arrays of size (pixel_coord1_slice.shape[0],pixel_coord1_slice.shape[1],np.amax(pixel_span_slice),np.amax(pixel_span_slice))
    max_span=(np.amax(np.ceil(pixel_span_slice))+1).astype('int32')
    
    map_coord1_slices=np.zeros(pixel_coord1_slice.shape+(max_span, max_span, ),dtype='int32')
    map_coord2_slices=np.zeros(pixel_coord2_slice.shape+(max_span, max_span, ),dtype='int32')
    geometrical_weighting_slices=np.zeros(pixel_coord1_slice.shape+(max_span, max_span, ),dtype='float32')
    
    start_pixel_coord1_slice=np.floor(pixel_coord1_slice + 0.5 - pixel_span_slice/2).astype('int32')
    start_pixel_coord2_slice=np.floor(pixel_coord2_slice + 0.5 - pixel_span_slice/2).astype('int32')
    
    for map_coord1_step in range(max_span):
        current_map_coord1=start_pixel_coord1_slice+map_coord1_step
        map_coord1_slices[:,:,map_coord1_step,:]=tl.tile(current_map_coord1, (1, 1, max_span))
        w=1-np.clip(pixel_coord1_slice-pixel_span_slice/2-current_map_coord1+0.5,0,1)-np.clip(current_map_coord1+0.5-(pixel_coord1_slice+pixel_span_slice/2),0,1)
        geometrical_weighting_slices[:,:,map_coord1_step,:]=tl.tile(w, (1, 1, max_span))

    for map_coord2_step in range(max_span):
        current_map_coord2=start_pixel_coord2_slice+map_coord2_step
        map_coord2_slices[:,:,:,map_coord2_step]=tl.tile(current_map_coord2, (1, 1, max_span))
        w=1-np.clip(pixel_coord2_slice-pixel_span_slice/2-current_map_coord2+0.5,0,1)-np.clip(current_map_coord2+0.5-(pixel_coord2_slice+pixel_span_slice/2),0,1)
        geometrical_weighting_slices[:,:,:,map_coord2_step]=geometrical_weighting_slices[:,:,:,map_coord2_step]*tl.tile(w, (1, 1, max_span))
            
    return map_coord1_slices, map_coord2_slices, geometrical_weighting_slices
    
def proj_mapper(x, y, w, val, proj_shape):
    xmin=-0.5
    xmax=proj_shape[0]-0.5
    ymin=-0.5
    ymax=proj_shape[1]-0.5
    w_vals=w*tl.tile(val,(1, 1, w.shape[2], w.shape[3]))
    hist, xedge, yedge=np.histogram2d(x.flatten(), y.flatten(), bins=np.array([xmax-xmin,ymax-ymin]), range=np.array([[xmin, xmax], [ymin, ymax]]), weights=w_vals.flatten())
    return hist

def proj_filters(proj,source_dist,voxel_size,source_pos,filter_type):
    coord2,coord1=np.meshgrid(np.arange(proj.shape[1])-source_pos[1],np.arange(proj.shape[0])-source_pos[0])
    pre_weight=source_dist/np.sqrt(source_dist**2+voxel_size**2*(coord1**2+coord2**2))
    if filter_type is 'cone_bck_pre_weight':
        proj=proj*pre_weight
    elif filter_type is 'cone_fwd_post_weight':
        proj=proj/pre_weight
    elif filter_type is 'fdk_prefilter':
        #old_proj=proj
        proj=proj*pre_weight
        proj=flat_panel_fdk_filter(proj,voxel_size)
        proj=proj.astype('float32')
        #pg.image(old_proj)
        #pg.image(proj)
        #stop
    else:
        print('unknown filter type')
        stop
    return proj
    
def radial_mask_tomo(tomo_shape):
    radial_slice=tl.radial_dist_map(tomo_shape[1:3],radius='edge')
    return np.tile(radial_slice,(tomo_shape[0],1,1))

def proj_std(tomo, dfstack, angles, source_dist, source_pos, rot_pos, det_dist, pixel_size):
    std_vector=np.zeros(angles.shape)
    for angle_no in range(len(angles)):
        angle=angles[angle_no]
        slice_nos=np.arange(dfstack.shape[0])  
        current_tomo_proj, map_coeff_sum=forward_proj(tomo, slice_nos, angle, source_dist, source_pos, rot_pos, det_dist, pixel_size)
        std_vector[angle_no]=np.sqrt(np.mean((dfstack[:,:,angle_no]-current_tomo_proj)**2))
    return std_vector
    
def plot_df_slice(img,colormap='rgb',img_title='or_df_slice',bck_color='k'):
    # assumes img to be the first order fourier component
    rgb_img=np.zeros((img.shape[0],img.shape[1],3),dtype='float32')
    phase=np.angle(img)
    norm=np.abs(img)
    colorsign=1
    if bck_color=='w':
        colorsign=-1
        rgb_img=rgb_img+np.max(norm)
    if colormap=='rgb':
        for color_no in range(3):
            rgb_img[:,:,color_no]+=colorsign*norm*(np.cos((phase+np.pi*color_no*2/3)/2)**2)
    elif colormap=='rg':
        for color_no in range(2):
            rgb_img[:,:,color_no]=colorsign*norm*(np.cos((phase+np.pi*color_no)/2)**2)
    '''
    if bck_color=='w':
        rgb_img=np.zeros((img.shape[0],img.shape[1],3),dtype='float32')+np.max(norm)
        if colormap=='rgb':
            for color_no in range(3):
                rgb_img[:,:,color_no]+=-norm*(np.cos((phase+np.pi*(color_no)*2/3)/2)**2)/2
        elif colormap=='rg':
            for color_no in range(2):
                rgb_img[:,:,color_no]=norm*(np.cos((phase+np.pi*color_no)/2)**2)
    '''
    h=pg.image(rgb_img,title=img_title)
    return h

def dtomo_scaler(dtomo, tomo, dfstack, angles, source_dist, source_pos, rot_pos, det_dist, pixel_size):
    #tl_ct_rec: Slå dSE_dtomo_scaler og proj_error-funktionerne sammen
    dtomo_proj_dot_res=0
    dtomo_proj_dot_dtomo_proj=0
    std_vector=np.zeros((angles.size,),dtype='float32')
    for angle_no in range(len(angles)):
        angle=angles[angle_no]
        slice_nos=np.arange(dfstack.shape[0])
        current_tomo_proj, map_coeff_sum=forward_proj(tomo, slice_nos, angle, source_dist, source_pos, rot_pos, det_dist, pixel_size)
        current_dtomo_proj, map_coeff_sum=forward_proj(dtomo, slice_nos, angle, source_dist, source_pos, rot_pos, det_dist, pixel_size)
        current_res_proj=dfstack[:,:,angle_no]-current_tomo_proj
        std_vector[angle_no]=np.sqrt(np.mean(current_res_proj**2))
        dtomo_proj_dot_res+=np.sum(current_dtomo_proj*current_res_proj)
        dtomo_proj_dot_dtomo_proj+=np.sum(current_dtomo_proj*current_dtomo_proj)
    dSE_dtomo_scale=dtomo_proj_dot_res/dtomo_proj_dot_dtomo_proj
    return dSE_dtomo_scale, std_vector
    
def voxel_similarizer(tomo):
    #assumes only even Fourier orders to be included in the tomogram.
    #Assumes tomogram to be 4D with 3 spatial dimensions first and fourier coefficients along the fourth dimension
    phi_tomo=np.angle(tomo[:,:,:,1])
    abs_tomo=np.sqrt(np.sum(np.abs(tomo)**2,(3,)))
    
    for ft_comp_no in np.arange(1,tomo.shape[3]):
        tomo[:,:,:,ft_comp_no]=tomo[:,:,:,ft_comp_no]*np.exp(-1j*phi_tomo*ft_comp_no)
    total_ft_comps=np.sum(tomo,axis=(0,1,2))

    total_ft_comps=np.real(total_ft_comps)
    total_ft_comps=pos_tomo(np.reshape(total_ft_comps,(1,1,1,total_ft_comps.size)),10*total_ft_comps.size)
    total_ft_comps=np.reshape(total_ft_comps,(total_ft_comps.size,))

    scale_tomo=abs_tomo/np.sqrt(np.sum(np.abs(total_ft_comps)**2))
    for ft_comp_no in np.arange(total_ft_comps.size):
        tomo[:,:,:,ft_comp_no]=scale_tomo*np.exp(1j*phi_tomo*ft_comp_no)*total_ft_comps[ft_comp_no]
    return tomo
    
def pos_tomo(tomo, no_of_angles):
    angles=np.linspace(0, np.pi, no_of_angles, endpoint=False)
    tomo[np.real(tomo[:,:,:,0])<0,0]=0#np.min(tomo[tomo[:,:,:,0]>0,0])
    min_tomo=np.zeros(tomo.shape[:-1],dtype='float32')
    for angle_no in range(no_of_angles):
        angle=angles[angle_no]
        eps_tomo=np.copy(np.real(tomo[:,:,:,0]))
        for ft_comp_no in np.arange(1,tomo.shape[3]):
            eps_tomo+=2*np.real(tomo[:,:,:,ft_comp_no]*np.exp(1j*2*angle*ft_comp_no))
        min_tomo=np.minimum(min_tomo, eps_tomo)
    higher_order_scaler=tomo[:,:,:,0]/(tomo[:,:,:,0]-min_tomo)
    higher_order_scaler[higher_order_scaler>1]=1
    higher_order_scaler[min_tomo==tomo[:,:,:,0]]=1
    higher_order_scaler[tomo[:,:,:,0]==0]=0
    for ft_comp_no in range(1,tomo.shape[3]):    
        tomo[:,:,:,ft_comp_no]=tomo[:,:,:,ft_comp_no]*higher_order_scaler
    return tomo
    
def flat_panel_fdk_filter(proj,voxel_size):
    # Zero-padding
    orig_proj_shape=proj.shape
    power_of_two=np.ceil(np.log(proj.shape[1])/np.log(2))+1
    new_width=2**power_of_two
    no_of_zeros=new_width-proj.shape[1]
    proj=np.concatenate((np.zeros((proj.shape[0],np.ceil(no_of_zeros/2))),proj,np.zeros((proj.shape[0],np.floor(no_of_zeros/2)))),axis=1)
    
    # Building filter
    real_space_filter=np.zeros(new_width)
    real_space_filter[0]=1/(4*voxel_size)
    for ind in np.arange(1,orig_proj_shape[1],2):
        real_space_filter[[ind,-ind]]=-1/((ind*np.pi)**2*voxel_size)
    
    ft_filter=np.real(np.fft.fft(real_space_filter))
    '''
    # Trying new filter
    f=fftfreq(int(new_width),d=voxel_size)
    ft_filter = 2*np.abs(f) 
    ####
    '''
    ramp_filter=np.tile(ft_filter,(proj.shape[0],1))

    # Filtering
    proj=np.fft.ifft(np.fft.fft(proj)*ramp_filter)
    proj=np.real(proj[:,np.ceil(no_of_zeros/2):-np.floor(no_of_zeros/2)])
    
    return proj
    
def ray_normalized_fdk(tomo, dfstack, angles, source_dist, source_pos, rot_pos, det_dist, pixel_size):
    unscaled_tomo=tomo/tl.tile(tomo[:,:,:,0],(1,1,1,tomo.shape[3]))
    unscaled_tomo[~np.isfinite(unscaled_tomo)]=0
    fdk_tomo=np.zeros((dfstack.shape[0],dfstack.shape[1],dfstack.shape[1]),dtype='float32')
    for proj_iter_no in np.arange(len(angles)):
        pixel_coord2_slice, pixel_span_slice, prop_angle_slice, mag_slice=hor_plane_voxel_coordinates(tomo.shape, angles[proj_iter_no], source_dist, source_pos, rot_pos, det_dist, pixel_size)
        current_tomo=ft_comp_collapser(unscaled_tomo,prop_angle_slice)
        unscaled_proj, map_coeff_sum=forward_proj(current_tomo, np.arange(tomo.shape[0]), angles[proj_iter_no], source_dist, source_pos, rot_pos, det_dist, pixel_size, map_area='disk')
        proj_scaler=unscaled_proj/map_coeff_sum
        proj_scaler[proj_scaler<0.2]=0.2
        proj_scaler[proj_scaler>5]=5
        fdk_tomo+=back_proj(dfstack[:,:,proj_iter_no]/proj_scaler, np.arange(dfstack.shape[0]), angles[proj_iter_no], source_dist, source_pos, rot_pos, det_dist, pixel_size,proj_filter='fdk')*2*np.pi/len(angles)
        if np.mod(proj_iter_no,np.round(angles.size/10)) == 1:
            print('fdk backprojecting angle no ' + str(proj_iter_no))
    tomo=unscaled_tomo*tl.tile(fdk_tomo,(1,1,1,tomo.shape[3]))
    return tomo

def k_punisher(tomo,k_step):
    for ft_comp_no in range(tomo.shape[3]):
        tomo[:,:,:,ft_comp_no]=np.exp(-ft_comp_no**2*k_step)*tomo[:,:,:,ft_comp_no]
    return tomo
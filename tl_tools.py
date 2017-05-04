# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 08:31:43 2015

@author: torstenl
"""

import numpy as np
import struct
import pyqtgraph as pg
import os
import dicom
from skimage import filters
from numpy import linalg

# Code examples
'''
pg.QtGui.QApplication.closeAllWindows()
import pyqtgraph.examples
pyqtgraph.examples.run()
'''

pi=np.pi

def tile(A, reps):
    #appends dimensions instead of prepending them as numpy.tile does:
    # if A.ndim!=np.size(reps) the smaller of the two is appended to make them match

    # First making sure reps is single-dimensional array:
    reps=np.reshape(np.asarray(reps),(np.size(reps),))
    
    # Then appending elements to reps or dimensions to A as necessary:
    if A.ndim<np.size(reps):
        A=np.reshape(A,A.shape+(1,)*(np.size(reps)-A.ndim))
    elif A.ndim>np.size(reps):
        reps=np.append(reps,np.ones((A.ndim-np.size(reps), )))
        
    # Finaly we can just use numpy built-in tile function:
    return np.tile(A,reps)

def radial_dist_map(map_shape,center='center',radius=None,mode='real'):
    # if radius is given, function returns a binary map of ones for distances to center smaller than radius
    # else it returns a distance map
    # if mode=='ft' or center=='ft_center' distances are given in units of frequency, and not "pixels"
    if isinstance(center,str):
        if center=='center':
            center=np.array([float(map_shape[0]-1)/2,float(map_shape[1]-1)/2])
        elif center=='ft_center':
            center=np.array([int(map_shape[0])/2,int(map_shape[1])/2])
            mode='ft'
    
    if mode=='ft':
        f_y,f_x=np.meshgrid(np.arange(map_shape[1]),np.arange(map_shape[0]))
        f_y=(f_y-center[1]).astype(float)
        f_x=(f_x-center[0]).astype(float)
        y=f_y/map_shape[1]
        x=f_x/map_shape[0]
    else:
        y,x=np.meshgrid(np.arange(map_shape[1]),np.arange(map_shape[0]))
        y=y.astype(float)-center[1]
        x=x.astype(float)-center[0]    
    dist_map=np.sqrt(x**2+y**2)

    if not radius==None:
        if radius=='edge':
            g=np.asarray(map_shape,dtype='float32')-center
            radius=np.min(np.concatenate((center,g)))        
        dist_map+=-radius
        dist_map[dist_map>=0]=0
        dist_map[dist_map<0]=1
    return dist_map

def angular_map(map_shape,center='center',mode='ft'):
    # gives a map of angular coordinates around center.
    # if mode=='ft', map is assumed to be a fourier map, but angles are in corresponding real space
    if center=='center':
        center=np.array([float(map_shape[0]-1)/2,float(map_shape[1]-1)/2])
    elif center=='ft_center':
        center=np.array([int(map_shape[0])/2,int(map_shape[1])/2])
    if mode=='ft':
        f_y,f_x=np.meshgrid(np.arange(map_shape[1]),np.arange(map_shape[0]))
        f_y=(f_y-center[1]).astype(float)
        f_x=(f_x-center[0]).astype(float)
        y=f_y/map_shape[1]
        x=f_x/map_shape[0]
    else:
        y,x=np.meshgrid(np.arange(map_shape[1]),np.arange(map_shape[0])).astype(float)
        y=y-center[1]
        x=x-center[0]
    angle_map=np.arctan2(x,y)
    return angle_map

def angular_fourier_mask(img,mask_lims,std,mirror=False):
    from scipy.stats import norm
    # mask_lims: angular interval(s) to be masked out
    # first axis: (start_lim,end_lim)
    # second axis: interval no.
    # std is a vector of standard deviations associated with the mask intervals
    # len(std)=mask_lims.shape[1] or len(std)=1
    # if len(std)=0 std is the same for all mask intervals
    if mirror is not False:
        new_lims=np.zeros((2,2*mask_lims.shape[1]),dtype=mask_lims.dtype)
        new_lims[:,:mask_lims.shape[1]]=mask_lims
        new_lims[:,mask_lims.shape[1]:]=mask_lims+pi
        mask_lims=new_lims
    if isinstance(std,(int,float)):
        std=np.ones((mask_lims.shape[1],))*std
    ft_img=np.fft.fftshift(np.fft.fft2(img))
    ang_map=angular_map(ft_img.shape,center='ft_center',mode='ft')
    mask=np.ones(ft_img.shape,dtype=float)
    for mask_int_no in range(mask_lims.shape[1]):
        mask_width=np.mod(mask_lims[1,mask_int_no]-mask_lims[0,mask_int_no],2*pi)
        mask_center=mask_lims[0,mask_int_no]+mask_width/2
        mask_shift=pi-mask_center
        relative_angle=np.mod(ang_map+mask_shift,2*pi)-mask_shift
        temp_mask=1-norm.cdf(relative_angle,mask_lims[1,mask_int_no],std[mask_int_no])
        temp_mask=temp_mask*norm.cdf(relative_angle,mask_lims[0,mask_int_no],std[mask_int_no])
        mask=mask*(1-temp_mask)
    if np.issubdtype(img.dtype, float):
        return np.real(np.fft.ifft2(np.fft.ifftshift(mask*ft_img)))
    if np.issubdtype(img.dtype, complex):
        return np.fft.ifft2(np.fft.ifftshift(mask*ft_img))

def low_order_img(img,max_order,padding=None):
    # max_order: frequency threshold in units of 1./pixel
    # assumes img to be 2D or 3D. Performs high order suppression on first two dimensions
    img_shape=img.shape
    if len(img_shape)==2:
        img=np.reshape(img,img.shape+(1,))
    if padding:
        pad_mask=np.ones(img.shape,dtype=float)
        pad_mask=np.pad(pad_mask,((padding,padding),(padding,padding),(0,0)),'constant')
        img=np.pad(img,((padding,padding),(padding,padding),(0,0)),'constant')
    ft_mask=np.fft.ifftshift(radial_dist_map(img.shape[:2],center='ft_center',radius=max_order))
    filtered_img=np.fft.ifft2(np.fft.fft2(img,axes=(0,1))*tile(ft_mask,(1,1,img.shape[2])),axes=(0,1))
    if not np.issubdtype(img.dtype, 'complex'):
        filtered_img=np.real(filtered_img)
    if padding:
        filtered_pad_mask=np.fft.ifft2(np.fft.fft2(pad_mask,axes=(0,1))*tile(ft_mask,(1,1,img.shape[2])),axes=(0,1))
        if not np.issubdtype(img.dtype, 'complex'):
            filtered_pad_mask=np.real(filtered_pad_mask)
        filtered_img=filtered_img[padding:-padding,padding:-padding,:]/filtered_pad_mask[padding:-padding,padding:-padding,:]
    if len(img_shape)==2:
        filtered_img=np.reshape(filtered_img,img_shape)
    return filtered_img

def float_from_string(string):
    out_array=np.zeros(len(string)/4,dtype='float32')
    for n in range(0,len(string),4):
        string_bite=string[n:n+4]
        flo=struct.unpack('f',string_bite)
        flo=flo[0]
        out_array[n/4]=flo
    return out_array

def from_string(string,out_format,endian=''):
    # endian: sign describing endianness, '>' for big, '<' for little
    print(string[0])
    if out_format=='uint8':
        out_array=np.zeros(len(string),dtype='uint8')
        for n in range(0,len(string)):
            string_bite=string[n]
            flo=struct.unpack('B',string_bite)
            flo=flo[0]
            out_array[n]=flo
    if out_format=='float32':
        out_array=np.zeros(len(string)/4,dtype='float32')
        for n in range(0,len(string),4):
            string_bite=string[n:n+4]
            flo=struct.unpack(endian+'f',string_bite)
            flo=flo[0]
            out_array[n/4]=flo
    return out_array

def bad_pixel_corrector(array,std,cutoff_val=-np.inf):
    # Estimates correct values of bad pixels by gaussianly smearing the neighbour pixels
    # std gives the standard deviation of the gaussian distribution used for smearing.
    # std can be a tuple if std is not the same along all dimensions
    bad_pixel_mask=~np.isfinite(array)
    array[bad_pixel_mask]=-np.inf
    bad_pixel_mask[array<=cutoff_val]=True
    array[bad_pixel_mask]=0
    smeared_positive_mask=filters.gaussian(~bad_pixel_mask,std)
    smeared_array=filters.gaussian(array,std)/smeared_positive_mask
    array[bad_pixel_mask]=smeared_array[bad_pixel_mask]

def unpack_string(string,unpack_format):
    #unpack format according to struct package, e.g. '<f' for little endian 32-bit float
    chunk_size=struct.calcsize(unpack_format)
    out_array=np.zeros(len(string)/chunk_size,dtype=unpack_format)
    for n in range(0,len(string),chunk_size):
        string_bite=string[n:n+chunk_size]
        flo=struct.unpack(unpack_format,string_bite)
        flo=flo[0]
        out_array[n/chunk_size]=flo
    return out_array

def projloader(proj_no,filepath_and_name,projsize):
    f = open(filepath_and_name+str(proj_no).zfill(4)+'.raw','r')
    img_str=f.read(projsize[0]*projsize[1]*4)
    f.close()
    load_img=float_from_string(img_str)
    load_img.shape=(projsize[1],projsize[0])
    load_img=np.transpose(load_img)
    return load_img
    
def sliceloader(filepath_and_name,slice_no,slice_size,dtype='float32'):
    if dtype=='float32':
        f=open(filepath_and_name,'r')
        f.seek(slice_no*np.prod(slice_size)*4)
        load_slice=f.read(np.prod(slice_size)*4)
        load_slice=float_from_string(load_slice)
        load_slice.shape=slice_size
    return load_slice

def multix_line_adder(loadname):
    no_of_energy_bins=np.fromfile(loadname,dtype='uint32')[10]
    no_of_pixels=np.fromfile(loadname,dtype='uint32')[11]
    integration_time=np.fromfile(loadname,dtype='uint32')[12]
    no_of_readouts=np.fromfile(loadname,dtype='uint32')[15]
    lines=np.fromfile(loadname,dtype='uint16')[256:]
    lines.shape=(no_of_readouts,no_of_pixels,no_of_energy_bins)
    line=np.sum(lines,axis=0)
    energy_spectrum=np.arange(20,160,float(140)/no_of_energy_bins)
    time=integration_time*no_of_readouts
    return line, energy_spectrum, time

def dcm_stack_loader(PathDicom):
    #loads all .dcm images in PathDicom and orders them according to decending SliceLocation
    #SliceLocation duplicates are ignored
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))
    slice_pos_vector=[]
    for filenameDCM in lstFilesDCM:
        ds = dicom.read_file(filenameDCM,force=True)
        slice_pos_vector.append(ds.SliceLocation)
    slice_pos_vector=list(set(slice_pos_vector))
    slice_pos_vector.sort(reverse=True)
    RefDs = dicom.read_file(lstFilesDCM[0],force=True)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(slice_pos_vector))
    voxeldims = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
    tomo = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    for filenameDCM in lstFilesDCM:
        ds = dicom.read_file(filenameDCM,force=True)
        tomo[:, :, slice_pos_vector.index(ds.SliceLocation)] = ds.pixel_array  
    return tomo, voxeldims
    
def multix_mu_maker(loaddir,sample_files,flat_file):
    #directory to look for files
    #sample_files: tuple of strings giving name of sample files
    #flat_file: string giving flatfield filename
    flat_line, energy_spectrum, time=multix_line_adder(loaddir+flat_file)
    flat_line=np.ndarray.astype(flat_line,dtype='float64')
    mu_map=np.zeros(flat_line.shape+(len(sample_files),),dtype='float64')
    std_map=np.zeros(flat_line.shape+(len(sample_files),),dtype='float64')
    for sample_line_no in range(len(sample_files)):
        line, energy_spectrum, time=multix_line_adder(loaddir+sample_files[sample_line_no])
        line=np.ndarray.astype(line,dtype='float64')
        mu_map[:,:,sample_line_no]=-np.log(line/flat_line)
        std_map[:,:,sample_line_no]=np.sqrt(1/flat_line+1/line)
    return mu_map, std_map, energy_spectrum

def pg_close():
    #super().pg.QtGui.QApplication.closeAllWindows()
    parentClass.__init__.pg.QtGui.QApplication.closeAllWindows()

def pag_filter(T_stack,d,delta,beta,photon_energy,pixelsize,zero_padding,validity_cutoff=1):
    # all distance measures in meters
    # energy in keV
    # validity-cutoff=1 corresponds to a phase contribution of pi/18 (i.e. "10 degrees"), giving an error of 1.5 %
    # if no validity-cutoff is desired, set it to inf
    wave_length=12.398e-10/photon_energy
    wave_number=2*np.pi/wave_length
    mu=2*wave_number*beta
    T_pad=np.pad(T_stack,((zero_padding,zero_padding),(zero_padding,zero_padding),(0,0)),'constant')
    pad_mask=np.ones(T_stack.shape[0:2],dtype=float)
    pad_mask=np.pad(pad_mask,zero_padding,'constant')
    # calculating validity limit of TIE    
    f_lim=np.sqrt(wave_number/(36*np.pi*d))*validity_cutoff
    # Building filter:
    f2=squared_spatial_frequency_map(T_pad.shape[0:2],pixelsize)
    validity_filter=f2<f_lim**2
    h=validity_filter/(1+d*delta/mu*((2*np.pi)**2)*f2)
    # filtering transmission image
    ft_T_pad=np.fft.fft2(T_pad,axes=(0,1))
    filtered_ft_T_pad=ft_T_pad*tile(h,(1,1,T_pad.shape[2]))
    filtered_T=np.real(np.fft.ifft2(filtered_ft_T_pad,axes=(0,1)))
    # filtering padding mask
    ft_pad_mask=np.fft.fft2(pad_mask)
    filtered_ft_pad_mask=ft_pad_mask*h
    filtered_pad_mask=np.real(np.fft.ifft2(filtered_ft_pad_mask))
    # normalising with filtered_pad_mask:
    T_out=filtered_T/tile(filtered_pad_mask,(1,1,T_pad.shape[2]))
    return T_out[zero_padding:-zero_padding,zero_padding:-zero_padding,:]
    
def fresnel_propagator(psi_stack,k,d,pixelsize,edge_padding=50):
    # the propagator disregards the plane wave phase factor e^{ikd}
    if len(psi_stack.shape)==2:
        psi_stack=np.reshape(psi_stack,psi_stack.shape+(1,))
        orig_dim=2
    elif len(psi_stack.shape)==3:
        orig_dim=3
    else:
        print 'dimensionality of wavefunction array not supported. Must be 2D or 3D'
        stop
    if edge_padding=='none':
        psi_pad=psi_stack
    else:
        psi_pad=np.pad(psi_stack,((edge_padding,edge_padding),(edge_padding,edge_padding),(0,0)),mode='edge')
    f2=squared_spatial_frequency_map(psi_pad.shape[0:2],pixelsize)
    h=np.exp(1j*2*(np.pi)**2*f2*d/k)
    ft_psi_pad=np.fft.fft2(psi_pad,axes=(0,1))
    psi_pad_d=np.fft.ifft2(ft_psi_pad*tile(h,(1,1,psi_pad.shape[2])),axes=(0,1))
    if orig_dim==3:
        if edge_padding=='none':
            return psi_pad_d
        else:
            return psi_pad_d[edge_padding:-edge_padding,edge_padding:-edge_padding,:]
    elif orig_dim==2:
        if edge_padding=='none':
            return psi_pad_d
        else:
            return psi_pad_d[edge_padding:-edge_padding,edge_padding:-edge_padding]

def squared_spatial_frequency_map(det_shape,pixelsize):
    # assumes a 2d detector screen with square pixels
    x_coords=np.arange(0,det_shape[0])
    x_coords=np.minimum(x_coords,det_shape[0]-x_coords)
    y_coords=np.arange(0,det_shape[1])
    y_coords=np.minimum(y_coords,det_shape[1]-y_coords)
    fx,fy=np.meshgrid(x_coords,y_coords,indexing='ij')
    f2=(fx/(pixelsize*x_coords.size))**2+(fy/(pixelsize*y_coords.size))**2
    return f2

def xradia_bin_slice_collector(path):
    import linecache
    tomo_size=np.zeros((3,),dtype=int)
    for dim_no in range(3):
        tomo_size[dim_no]=int(linecache.getline(path+'Header.txt', dim_no+2)[-5:-1])
    if os.path.exists(path+'tomo.vol'):
        os.remove(path+'tomo.vol')
    for slice_no in range(tomo_size[0]):
        imslice=np.reshape(np.fromfile(path+str(slice_no+1).zfill(4)+'.bin',dtype='float32'),tomo_size[1:3])
        f=open(path+'tomo.vol','a')
        f.write(imslice)
        f.close()
        print slice_no

def tomo_binner(tomo,bin_factor,method='mean'):
    bin_dims=np.floor(np.array(tomo.shape)/bin_factor)
    tomo=tomo[:bin_dims[0]*bin_factor,:bin_dims[1]*bin_factor,:bin_dims[2]*bin_factor]
    tomo=np.reshape(tomo,(bin_dims[0],bin_factor,bin_dims[1],bin_factor,bin_dims[2],bin_factor),order='C')
    tomo=np.transpose(tomo,axes=(0,2,4,1,3,5))
    tomo=np.reshape(tomo,tuple(bin_dims)+(bin_factor**3,))
    if method=='mean':
        tomo=np.mean(tomo,axis=3)
    return tomo

def img_binner(img,bin_factor,method='mean'):
    bin_dims=np.floor(np.array(img.shape)/bin_factor)
    img=img[:bin_dims[0]*bin_factor,:bin_dims[1]*bin_factor]
    img=np.reshape(img,(bin_dims[0],bin_factor,bin_dims[1],bin_factor),order='C')
    img=np.transpose(img,axes=(0,2,1,3))
    img=np.reshape(img,tuple(bin_dims)+(bin_factor**2,))
    if method=='mean':
        img=np.mean(img,axis=2)
    return img

def line_profile(img,startpoint,endpoint,step_length=1):
    # currently only implemented for 2D images    
    from scipy import interpolate

    axis_vectors=(np.arange(img.shape[0]),np.arange(img.shape[1]))    

    startpoint=np.asarray(startpoint)    
    endpoint=np.asarray(endpoint)
    line_length=np.sqrt(np.sum((startpoint-endpoint)**2))
    
    line_axis=np.arange(0,line_length,step_length)

    gradient_vector=(endpoint-startpoint)/line_length
    coord_matrix=np.dot(np.reshape(gradient_vector,(2,1)),np.reshape(line_axis,(1,len(line_axis))))
    coord_matrix=coord_matrix+tile(startpoint,(1,coord_matrix.shape[1]))
    profile_line=interpolate.interpn(axis_vectors, img, np.transpose(coord_matrix), method='linear', bounds_error=True,)
    return profile_line, line_axis

def slicing(arr,dim,indexes):
    shifted_arr=np.moveaxis(arr,dim,0)
    sliced_shifted_arr=shifted_arr[indexes,]
    sliced_array=np.moveaxis(sliced_shifted_arr,0,dim)
    return sliced_array

def discrete_derivative(arr,dim,step_length=1):
    shifted_arr=np.moveaxis(arr,dim,0)
    shifted_derivative=np.zeros(shifted_arr.shape)
    shifted_derivative[1:-1,]=(shifted_arr[2:,]-shifted_arr[:-2,])/(2*step_length)
    shifted_derivative[0,]=(shifted_arr[1,]-shifted_arr[0,])/step_length
    shifted_derivative[-1,]=(shifted_arr[-1,]-shifted_arr[-2,])/step_length
    return np.moveaxis(shifted_derivative,0,dim)

def nabla(arr,step_length=1):
    nabla_arr=np.zeros((len(arr.shape),)+arr.shape)
    for dim in range(len(arr.shape)):
        nabla_arr[dim,]=discrete_derivative(arr,dim,step_length)
    return nabla_arr

def structure_tensor(arr,sigma,step_length=1):
    #sigma is in physical length, i.e. if step_length=3 and sigma=2, the gausian filter has a standard deviation of 2/3 pixels
    tensor_arr=np.zeros((len(arr.shape),len(arr.shape),)+arr.shape,dtype=float)
    nabla_arr=nabla(arr)
    for dir1 in range(len(arr.shape)):
        for dir2 in range(len(arr.shape)):
            tensor_arr[dir1,dir2,]=filters.gaussian(nabla_arr[dir1,]*nabla_arr[dir2,], np.asarray(sigma)/step_length)
    return tensor_arr/step_length**2 #Dividing by step_length**2 to take physical length into account (this was not done when calling nabla)

def directional_structure(structure_tensor,v):
    #Assumes v to be a one-dimensional vector v.shape=(n,)
    return np.sum(structure_tensor*tile(v,(1,)+structure_tensor.shape[1:]),axis=0)

def dim_collapser(a,coord,dim):
    # a: starting array
    # coord: array of coordinates of the desired values along dimension dim.
    # Last dims of coord has same shape as a, except for dimension dim, where coord.shape[dim]=1
    # If dims of a extend beyond dims of coord (plus "dim"), remaining dimensions are prepended to coord
    # dim: dimension along which to pick the desired values from a
    if len(a.shape)>len(coord.shape)+1:
        no_of_missing_dims=len(a.shape)-1-len(coord.shape)
        coord=np.tile(coord,a.shape[:no_of_missing_dims]+tuple(np.ones(len(coord.shape))))
    dim_length=a.shape[dim]
    a_m=np.moveaxis(a,dim,0).reshape((dim_length,a.size/dim_length))
    coord_m=np.moveaxis(coord,dim,0).reshape((1,a.size/dim_length))
    collapsed_a_m=a_m[coord_m,np.arange(a.size/dim_length)]
    collapsed_a_t_shape=np.moveaxis(a,dim,0)[0:1,:].shape
    collapsed_a_t=collapsed_a_m.reshape(collapsed_a_t_shape)
    collapsed_a=np.moveaxis(collapsed_a_t,0,dim)
    return collapsed_a
'''
def vector_orderer(vector_priority,vector_stack):
    # Assumes first dim of vector_stack to be the vectors,
    # second dim to be the different vectors among which to chose
    # all remaining dimensions are just coordinates over which the ordering is performed
    # returns an array of vectors in order of decreasing priority and an array of the priorities in corresponding order
    vector_out=np.zeros(vector_stack.shape,dtype=float)
    priority_out=np.zeros(vector_priority.shape,dtype=float)
    for vector_no in range(vector_priority.shape[0]):
        current_max_coord=np.argmax(vector_priority,0)
        priority_out[vector_no,]=current_max_coord
        print vector_no
        for vector_coord_no in range(vector_stack.shape[0]):
            vector_out[:,vector_no,]=vector_stack[:,current_max_coord,]
        vector_priority[current_max_coord,]=0
    return priority_out,vector_out
#'''
    
def matrix_orientations(matrix_stack):
    #matrix_dimensions assumed to be the first two dimensions in matrix_stack. matrix_stack can have any dimensionality larger than or equal to 2
    shifted_stack=np.moveaxis(matrix_stack,0,-1)
    shifted_stack=np.moveaxis(shifted_stack,0,-1)
    print shifted_stack.shape
    eig_values,eig_vectors=linalg.eig(shifted_stack)
    eig_values=np.moveaxis(eig_values,-1,0)
    eig_vectors=np.moveaxis(eig_vectors,-1,0)
    eig_vectors=np.moveaxis(eig_vectors,-1,0)
    return eig_values,eig_vectors

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 
    
def imread(filename):
    #import string
    #dot_ind=string.rfind(filename,'.')
    dot_ind=filename.rfind('.')
    suffix=filename[dot_ind+1:]
    if suffix=='tif':
        import skimage.io
        img=skimage.io.imread(filename)
    if suffix=='tiff':
        import skimage.io
        img=skimage.io.imread(filename)
    return img

def imstack_loader(prefix,file_nos,no_of_digits,suffix):
    # loading first image
    img=imread(prefix+str(file_nos[0]).zfill(no_of_digits)+suffix)
    imstack=np.zeros(img.shape+(len(file_nos),),dtype=img.dtype)
    imstack[:,:,0]=img
    for load_no in range(1,len(file_nos)):
        imstack[:,:,load_no]=imread(prefix+str(file_nos[load_no]).zfill(no_of_digits)+suffix)
    return imstack

def read_picoimage(PathName,FileName):
    
    f = open(PathName+FileName,'r')
    
    # data in the file header are stored in Big Endian format ('b' directive)
    
    # read the general file header
    
    #version = from_string(f.read(4),'float32',endian='>')
    
    version = struct.unpack('>f',f.read(4))
    scantype = struct.unpack('>H',f.read(2))
    calcdomain = struct.unpack('>H',f.read(2))
    calctype = struct.unpack('>H',f.read(2))
    result = struct.unpack('>H',f.read(2))
    axeslist = unpack_string(f.read(101),'c')
    axesunits = unpack_string(f.read(53),'c')
    skipped_chunk=unpack_string(f.read(94),'B')
    
    # read information about axis 1 (step axis in raster scan)
    axis1index = struct.unpack('>l',f.read(4))
    skipped_chunk=unpack_string(f.read(12),'B')
    axis1startpos = unpack_string(f.read(4),'>f')/2.54
    axis1numpixels = struct.unpack('>L',f.read(4))[0]
    axis1pixelsize = unpack_string(f.read(4),'>f')/2.54
    axis1velocity = unpack_string(f.read(4),'>f')/2.54
    
    # read information about axis 2 (scan axis in raster scan)
    axis2index = struct.unpack('>l',f.read(4))
    skipped_chunk=unpack_string(f.read(12),'B')
    axis2startpos = unpack_string(f.read(4),'>f')/2.54
    axis2numpixels = struct.unpack('>L',f.read(4))[0]
    axis2pixelsize = unpack_string(f.read(4),'>f')/2.54
    axis2velocity = unpack_string(f.read(4),'>f')/2.54
    skipped_chunk=unpack_string(f.read(128),'B')
    
    # read acquisition details
    acquisitionmethod = unpack_string(f.read(16),'c')
    windowsize = struct.unpack('>f',f.read(4))
    acquisitionrate = struct.unpack('>f',f.read(4))
    conversionmultiplier = struct.unpack('>f',f.read(4))
    conversionoffset = struct.unpack('>f',f.read(4))
    pointsperwaveform = struct.unpack('>l',f.read(4))[0]
    timebetweenpoints = struct.unpack('>f',f.read(4))
    
    # empty space in picoimage file
    skipped_chunk=unpack_string(f.read(216),'B')
    skipped_chunk=unpack_string(f.read(1340),'B')
        
    # now follows information on all waveforms
    
    # data format changes to Little Endian ('l' directive)
    
    wf = np.zeros((axis1numpixels,axis2numpixels,pointsperwaveform),dtype='h') # preallocation of array of waveform vectors
    
    for ii in range(axis1numpixels):
        for jj in range(axis2numpixels):
            skipped_chunk=unpack_string(f.read(32),'B')
            t0 = struct.unpack('>f',f.read(4)) # in case the start time is needed
            skipped_chunk=unpack_string(f.read(92),'B')
            wf[ii,jj,:] = unpack_string(f.read(2*pointsperwaveform),'h')
    
    # complete picoimage file is now in memory
    return wf,axis1numpixels,axis1pixelsize,axis2numpixels,axis2pixelsize,pointsperwaveform,timebetweenpoints

def read_pmf(filename,shape,datatype='uint32'):
    img=np.fromfile(filename,dtype=datatype)
    print(datatype)
    print(img.shape)
    img.shape=shape
    #pg.image(img)
    tiles=np.asarray(img.shape)/256
    new_img=np.zeros((img.shape[0]+3*tiles[0]-3,img.shape[1]+3*tiles[1]-3),dtype=datatype)
    for tile_x in range(tiles[0]):
        for tile_y in range(tiles[1]):
            new_img[tile_x*259:(tile_x+1)*259-3,tile_y*259:(tile_y+1)*259-3]=img[tile_x*256:(tile_x+1)*256,tile_y*256:(tile_y+1)*256]
    #pg.image(new_img)
    
    for tile_edge_x in range(tiles[0]-1):
        new_img[(tile_edge_x+1)*259-4,:]=new_img[(tile_edge_x+1)*259-4,:]/2.5
        new_img[(tile_edge_x+1)*259,:]=new_img[(tile_edge_x+1)*259,:]/2.5
        new_img[(tile_edge_x+1)*259-3,:]=new_img[(tile_edge_x+1)*259-4,:]
        new_img[(tile_edge_x+1)*259-2,:]=new_img[(tile_edge_x+1)*259-4,:]/2+new_img[(tile_edge_x+1)*259,:]/2
        new_img[(tile_edge_x+1)*259-1,:]=new_img[(tile_edge_x+1)*259,:]
    #pg.image(new_img)
    
    for tile_edge_y in range(tiles[1]-1):
        line_no=(tile_edge_y+1)*259
        for tile_corner_x in range(tiles[0]-1):
            x_line_no=(tile_corner_x+1)*259
            new_img[x_line_no-4:x_line_no+1,line_no-4]=new_img[x_line_no-4:x_line_no+1,line_no-4]*2.5
            new_img[x_line_no-4:x_line_no+1,line_no]=new_img[x_line_no-4:x_line_no+1,line_no]*2.5
        
        new_img[:,line_no-4]=new_img[:,line_no-4]/2.5
        new_img[:,line_no]=new_img[:,line_no]/2.5
        new_img[:,line_no-3]=new_img[:,line_no-4]
        new_img[:,line_no-2]=new_img[:,line_no-4]/2+new_img[:,line_no]/2
        new_img[:,line_no-1]=new_img[:,line_no]
    
    return(new_img)
    
    
    
    
# tl_tools

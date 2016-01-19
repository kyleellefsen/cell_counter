# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:58:17 2015

@author: Kyle Ellefsen


Use this file to loop through all the files, extract the cell locations, and save the output.  

How to use:
1) Install Github Desktop, log in with your github account, and clone Flika into github.  Flika is located at https://github.com/kyleellefsen/Flika
2) Open the cell_counter.py file and edit the path_to_github variable to the correct directory (wherever all your github projects are located)
3) In this file, edit the 'path_to_currentFile' variable to the directory where this file is located
4) In this file, edit the 'directory' variable to the directory where the images are located.
5) Run the file.  The results will be saved out as csv files.

"""
if __name__=='__main__':
    import sys, os
    import time
    import numpy as np
    from cell_counter import *
    from FLIKA import *; app = QApplication(sys.argv); initializeMainGui()
    path_to_currentFile=r'D:\Desktop\cell_counter' #__file__
    sys.path.insert(1,path_to_currentFile)

def extractPoints(image_location, parameters, displayResults=False):
    tic=time.time()
    mask_radius, thresh, density_thresh, center_minDensity, center_minDistance, gaussianblur_sigma, min_number_of_pixels_in_cell = parameters
    original=open_file(image_location)
    blurred=gaussian_blur(gaussianblur_sigma,keepSourceWindow=True)
    A=original.image-blurred.image
    close(blurred)
    A[:2*gaussianblur_sigma,:]=0
    A[-2*gaussianblur_sigma:,:]=0
    A[:,:2*gaussianblur_sigma]=0
    A[:,-2*gaussianblur_sigma:]=0
    A_norm=normalize(A)
    Densities=getDensities_multi(A_norm,thresh,mask_radius)
    if displayResults:
        Window(Densities,'Densities')
    higher_pts,idxs=getHigherPoints_multi(Densities,density_thresh)
    #pw=plot_higher_pts(higher_pts)
    clusters, clusters_idx=find_clusters(higher_pts,idxs,center_minDensity,center_minDistance)
    clusters=filter_bad_cells(clusters,min_number_of_pixels_in_cell)
    if displayResults:
        cluster_window=plot_clusters(clusters,A.shape)
    pts,flikapts_fname=getPoints(clusters,original.image,image_location,displayResults)
    
    if displayResults:
        original.setAsCurrentWindow()
        load_points(flikapts_fname)
        background(cluster_window,original,.2,True)
        close(cluster_window)
    if not displayResults:
        close(original)
    toc=time.time()-tic
    print('Time to process {}: {} s'.format(image_location, toc))
    return pts


###############################################################################
#                    CONSTANTS
###############################################################################
mask_radius=19
thresh=.5
density_thresh=50
center_minDensity=0
center_minDistance=8
gaussianblur_sigma=20
min_number_of_pixels_in_cell=50
displayResults=False
parameters=mask_radius, thresh, density_thresh, center_minDensity, center_minDistance, gaussianblur_sigma, min_number_of_pixels_in_cell
###############################################################################
###############################################################################


if __name__=='__main__':
    directory=r'D:\Desktop\cell_counter\test_files'
    images=[f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f)) and os.path.splitext(f)[1]=='.tif']
    #images=['1b_01.tif']
    for image in images:
        image_location=os.path.join(directory,image)
        pts=extractPoints(image_location, parameters, displayResults=False)
        o_filename=os.path.splitext(image_location)[0]+'.csv'
        np.savetxt(o_filename,pts, header='X position,Y position,Mean Amplitude', fmt='%.4f', delimiter=',', comments='')















# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:58:53 2015

@author: Kyle Ellefsen
"""
from __future__ import division
import os, sys
import numpy as np
from scipy import spatial
import matplotlib
import time
from skimage.measure import regionprops
from skimage.measure import label
from multiprocessing import cpu_count
import pyqtgraph as pg
from PyQt4.QtGui import QBrush
from PyQt4.QtCore import Qt



path_to_github=r'C:\Users\Kyle Ellefsen\Documents\GitHub'
sys.path.insert(1,os.path.join(path_to_github,'Flika'));
from process.progress_bar import ProgressBar
from FLIKA import *



















def getMask(nx=5,ny=5):
    mask=np.zeros((nx,ny))
    center=np.array([(nx-1)/2, (ny-1)/2]).astype(np.int)
    x0,y0=center
    for x in np.arange(nx):
        for y in np.arange(ny):
            if  ((x-x0)**2) / (x0**2)   +   ((y-y0)**2) / (y0**2) <= 1:
                mask[x,y]=1
    return mask, center
    
def normalize(A):
    A=np.copy(A)
    std=np.std(A)
    mean=np.mean(A)
    A-=mean
    A/=std
    return A
    
class Point():
    def __init__(self,idx):
        self.children=[]
        self.idx=idx
    def __repr__(self):
        return str(self.idx)
    def getDescendants(self):
        self.descendants=self.children[:]
        for child in self.children:
            self.descendants.extend(child.getDescendants())
        return self.descendants
        

        


    
###############################################################################
#                    Density
###############################################################################
def getDensities_multi(Image,thresh,mask_radius):
    if mask_radius%2==0:
        mask_radius+=1
    nCores = cpu_count()
    pxls=np.array(np.where(Image>thresh)).T
    block_ends=np.linspace(0,len(pxls),nCores+1).astype(np.int)
    data_blocks=[pxls[block_ends[i]:block_ends[i+1]] for i in np.arange(nCores)]
    args=(Image,mask_radius)
    progress = ProgressBar(calcDensity, data_blocks, args, nCores, msg='Calculating Density')
    if progress.results is None or any(r is None for r in progress.results):
        result=None
    else:
        result=np.sum(progress.results,0)
    return result
    
def calcDensity(q_results, q_progress, q_status, child_conn, args):
    pxls=child_conn.recv() # unfortunately this step takes a long time
    percent=0  # This is the variable we send back which displays our progress
    status=q_status.get(True) #this blocks the process from running until all processes are launched
    if status=='Stop':
        q_results.put(None) # if the user presses stop, return None


    ''' For testing: 
    Image=A_norm
    mx,my=Image.shape
    mask,center=getMask(mask_radius,mask_radius)
    mask_idx=np.where(mask)
    pxls=np.array(np.where(Image>thresh)).T
    i=0
    pxl=pxls[100000]
    x,y=pxl
    square_cutout=Image[x-center[0]:x+center[0]+1,y-center[1]:y+center[1]+1]
    cutout=square_cutout[mask_idx]
    '''
    
    
    Image,mask_radius=args #unpack all the variables inside the args tuple
    result=np.zeros(Image.shape)
    mx,my=Image.shape
    mask,center=getMask(mask_radius,mask_radius)
    mask_idx=np.where(mask)
    for i,pxl in enumerate(pxls):
        x,y=pxl
        square_cutout=Image[x-center[0]:x+center[0]+1,y-center[1]:y+center[1]+1]
        try:
            cutout=square_cutout[mask_idx]
        except IndexError:
            x0=x-center[0]
            xf=x+center[0]+1
            y0=y-center[1]
            yf=y+center[1]+1
            mask2=mask
            if x0<0:
                mask2=mask2[center[0]-x:,:]
                x0=0
            if y0<0:
                mask2=mask2[:,center[1]-y:]
                y0=0
            if xf>mx-1:
                mask2=mask2[:-(xf-mx+1),:]
                xf=mx-1
            if yf>my-1:
                mask2=mask2[:,:-(yf-my+1)]
                yf=my-1
            mask2_idx=np.where(mask2)
            square_cutout=Image[x0:xf,y0:yf]
            cutout=square_cutout[mask2_idx]
        pxl_val=Image[x,y]
        result[x,y]=np.sum(cutout[cutout<pxl_val]) 
        
        if percent<int(100*i/len(pxls)):
            percent=int(100*i/len(pxls))
            q_progress.put(percent+2) #I have no idea why the last two percent aren't displayed, but I'm adding 2 so it reaches 100
        if not q_status.empty(): #check if the stop button has been pressed
            stop=q_status.get(False)
            q_results.put(None)
            return                 
    # finally, when we've finished with our calculation, we send back the result
    q_results.put(result)



    
###############################################################################
#                    HIGHER POINTS
###############################################################################

def getHigherPoints_multi(Densities,density_thresh):
    """"
    STRUCTURE OF HIGHER_PTS:
    ['Distance to next highest point, index of higher point, value of current point']
    """
    
    nCores = cpu_count()
    mx,my=Densities.shape
    idxs=np.where(Densities>density_thresh)
    densities=Densities[idxs]
    densities_jittered=densities+np.arange(len(densities))/(2*np.float(len(densities))) #I do this so no two densities are the same, so each cluster has a peak.
    C     = np.zeros((mx,my) )
    C_idx=np.zeros((mx,my),dtype=np.int)
    idxs=np.vstack((idxs[0],idxs[1])).T
    C[idxs[:,0],idxs[:,1]]=densities_jittered
    C_idx[idxs[:,0],idxs[:,1]]=np.arange(len(idxs))
    print("Number of pixels to analyze: {}".format(len(idxs)))
    remander=np.arange(len(idxs))
    nTotal_pts=len(idxs)
    block_ends=np.linspace(0,len(remander),nCores+1).astype(np.int)
    data_blocks=[remander[block_ends[i]:block_ends[i+1]] for i in np.arange(nCores)]
    
    # create the ProgressBar object
    args=(nTotal_pts, C, idxs, densities_jittered, C_idx)
    progress = ProgressBar(getHigherPoint_multi, data_blocks, args, nCores, msg='Getting Higher Points')
    if progress.results is None or any(r is None for r in progress.results):
        higher_pts=None
    else:
        higher_pts=np.sum(progress.results,0)
    r=99
    maxDistance=np.sqrt(2*r**2)
    remander=np.argwhere(higher_pts[:,0]==0)
    remander=remander.T[0]
    if len(remander)==1:
        ii=remander[0]
        higher_pts[ii]=[maxDistance, ii, densities_jittered[ii]]
    elif len(remander)>1:
        dens2=densities_jittered[remander]
        idxs2=idxs[remander]
        D=spatial.distance_matrix(idxs2,idxs2)
        for pt in np.arange(len(D)):
            dd=D[pt,:]
            idx=np.argsort(dd)
            ii=1
            while dens2[idx[0]]>dens2[idx[ii]]: # We are searching for the closest point with a higher density
                ii+=1
                if ii==len(dd): #if this is the most dense point, then no point will have a higher density
                    higher_pts[remander[pt]]= [maxDistance, remander[pt], dens2[pt] ] 
                    break
            if ii!=len(dd): #if this is the most dense point, then no point will have a higher density
                higher_pts[remander[pt]]= [   dd[idx[ii]],  remander[idx[ii]],  dens2[idx[ii]]   ]
    return higher_pts, idxs


def getHigherPoint_multi(q_results, q_progress, q_status, child_conn, args):
    remander=child_conn.recv() # unfortunately this step takes a long time
    percent=0  # This is the variable we send back which displays our progress
    status=q_status.get(True) #this blocks the process from running until all processes are launched
    if status=='Stop':
        q_results.put(None) # if the user presses stop, return None
    nTotal_pts, C, idxs, densities_jittered, C_idx=args
    mx,my=C.shape
    higher_pts=np.zeros((nTotal_pts,3))
    for r in np.arange(5,99,2):
        mask,center=getMask(r,r)
        oldremander=remander
        remander=[]
        percent=0
        for loop_i, ii in enumerate(oldremander):
            if not q_status.empty(): #check if the stop button has been pressed
                stop=q_status.get(False)
                q_results.put(None)
                return
            if percent<int(100*loop_i/len(oldremander)):
                percent=int(100*loop_i/len(oldremander))
                q_progress.put(percent+2)
            idx=idxs[ii]
            density=densities_jittered[ii]
            x,y=idx
            center2=np.copy(center)
            x0=x-center[0]
            xf=x+center[0]+1
            y0=y-center[1]
            yf=y+center[1]+1
            mask2=np.copy(mask)
            if x0<0:
                mask2=mask2[center[0]-x:,:]
                center2[0]=x
                x0=0
            elif xf>mx:
                crop=-(xf-mx)
                if crop<0:
                   mask2=mask2[:crop,:]
                xf=mx
            if y0<0:
                mask2=mask2[:,center[1]-y:]
                center2[1]=y
                y0=0
            elif yf>my:
                crop=-(yf-my)
                if crop<0:
                   mask2=mask2[:,:crop]
                yf=my
                
            positions=np.array(np.where(mask2*C[x0:xf,y0:yf]>density)).astype(float).T-center2
            if len(positions)==0:
                remander.append(ii)
            else:
                distances=np.sqrt(positions[:,0]**2+positions[:,1]**2)
                higher_pt=positions[np.argmin(distances)].astype(np.int)+np.array([x0,y0])+center2
                higher_pt=C_idx[higher_pt[0],higher_pt[1]]
                higher_pt=[np.min(distances), higher_pt, density]
                higher_pts[ii]=higher_pt
    q_results.put(higher_pts)



def plot_higher_pts(higher_pts):
    higher_pts_tmp=higher_pts[higher_pts[:,0]>1]
    y=[d[0] for d in higher_pts_tmp] #smallest distance to higher point
    x=[d[2] for d in higher_pts_tmp] # density 
    pw=pg.PlotWidget()
    pw.plot(x,y,pen=None, symbolBrush=QBrush(Qt.blue), symbol='o')
    pw.plotItem.axes['left']['item'].setLabel('Smallest distance to denser point'); pw.plotItem.axes['bottom']['item'].setLabel('Density')
    pw.show()
    return pw

def find_clusters(higher_pts,idxs,center_minDensity,center_minDistance):
    centers=[]
    outsideROI=[]
    for i in np.arange(len(higher_pts)):
        y=higher_pts[i][0]#smallest distance to higher point
        x=higher_pts[i][2]# density 
        if x>center_minDensity and y>center_minDistance:
            centers.append(i)
        else:
            outsideROI.append(i)

    higher_pts2=higher_pts[:,1].astype(np.int) #contains index of next highest point
    points=[Point(i) for i in np.arange(len(higher_pts2))]
    loop=np.arange(len(higher_pts2))
    loop=np.delete(loop,centers)
    for i in loop:
        if higher_pts2[i]!=i:
            points[higher_pts2[i]].children.append(points[i])
    clusters=[]
    for center in centers:
        descendants=points[center].getDescendants()
        cluster=[d.idx for d in descendants]
        cluster=np.array(cluster+[center])
        clusters.append(cluster)
        
    clusters_idx=clusters
    clusters=[idxs[c] for c in clusters]
    return clusters, clusters_idx
    

def plot_clusters(clusters,imshape):
    mx,my=imshape
    cluster_im=np.zeros((mx,my,4))
    cmap=matplotlib.cm.gist_rainbow
    for i in np.arange(len(clusters)):
        color=cmap(int(((i%15)*255./16)))#+np.random.randint(255./12)))
        cluster=clusters[i]
        x,y=cluster.T
        cluster_im[x,y,:]=color
    return Window(cluster_im, 'Clusters')
    
def plot_cluster2(clusters):
    sizes=[np.max(cluster,0)-np.min(cluster,0) for cluster in clusters]
    np.max(sizes)
    mmx,mmy=np.max(sizes,0)+1
    Cells= np.zeros((len(clusters),mmx,mmy))
    for i,cluster in enumerate(clusters):
        x,y=(cluster-np.min(cluster,0)).T
        Cells[i,x,y]=1
    W=Window(Cells)
    return W
    
    
def filter_bad_cells(clusters,min_number_of_pixels_in_cell):
    new_clusters=[]
    for i, cluster in enumerate(clusters):
        if len(cluster)<min_number_of_pixels_in_cell:
            continue
        offset=np.min(cluster,0)
        cluster=cluster-offset
        mmx,mmy=np.max(cluster,0)+1
        image=np.zeros((mmx,mmy),dtype=np.uint8)
        x,y=cluster.T
        image[x,y]=1
        label_image=label(image,background=0)
        label_image+=1
        regions=regionprops(label_image)
        if len(regions)==1:
            new_clusters.append(cluster+offset)
        else:
            biggest_idx=np.argmax([region.area for region in regions])
            biggest=regionprops(label_image)[biggest_idx]
            if biggest.area>min_number_of_pixels_in_cell:
                new_clusters.append(biggest.coords+offset)
    return new_clusters


def getprops(clusters,image):
    properties=[]
    for i, cluster in enumerate(clusters):
        offset=np.min(cluster,0)
        cluster=cluster-offset
        mmx,mmy=np.max(cluster,0)+1
        im=np.zeros((mmx,mmy),dtype=np.uint8)
        x,y=cluster.T
        im[x,y]=1
        label_image=label(im,background=0)
        label_image+=1
        regions=regionprops(label_image)
        if len(regions)>1:
            print("ERROR You should have run this through 'filter_bad_cells' first" )
            return None
        r=regions[0]
        x=x+offset[0]
        y=y+offset[1]
        mean_vals=np.mean(image[x,y])
        meanx=np.mean(x)
        meany=np.mean(y)
        properties.append([i, meanx, meany, r.area, r.eccentricity, r.orientation, mean_vals, ])
    properties=np.array(properties)
    return properties


def getPoints(clusters,Image,image_location,saveFlikaPoints=True):
    mean_xs=[]; mean_ys=[]; mean_vals=[]
    for cluster in clusters:
        x,y=cluster.T
        vals=Image[x,y]
        mean_xs.append(np.sum(vals*x)/np.sum(vals))
        mean_ys.append(np.sum(vals*y)/np.sum(vals))
        mean_vals.append(np.mean(vals))
        
    pts = np.array([mean_xs,mean_ys,mean_vals]).T
    #pts=pts[pts[:,2]>400]
    if saveFlikaPoints:
        filename=os.path.splitext(image_location)[0]+'.txt'
        flika_pts=np.hstack((np.zeros((len(pts),1)),pts[:,:2]))
        np.savetxt(filename,flika_pts)
        return pts, filename
    else:
        return pts, None
    




if __name__ == '__main__':
    app = QApplication(sys.argv); initializeMainGui()
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
    image_location=r'D:\Desktop\cell_counter\test_files\1b_01_cropped.tif'
    ###############################################################################
    ###############################################################################

    original=open_file(image_location)
    blurred=gaussian_blur(gaussianblur_sigma,keepSourceWindow=True)
    
    A=original.image-blurred.image
    close(blurred)
    A[:2*gaussianblur_sigma,:]=0
    A[-2*gaussianblur_sigma:,:]=0
    A[:,:2*gaussianblur_sigma]=0
    A[:,-2*gaussianblur_sigma:]=0
    A_norm=normalize(A)
    mx,my=A.shape
    Densities=getDensities_multi(A_norm,thresh,mask_radius)
    Window(Densities,'Densities')
    higher_pts,idxs=getHigherPoints_multi(Densities,density_thresh)
    
    
    #pw=plot_higher_pts(higher_pts)
    clusters, clusters_idx=find_clusters(higher_pts,idxs, center_minDensity,center_minDistance)
    #plot_clusters(clusters)
    clusters=filter_bad_cells(clusters,min_number_of_pixels_in_cell)
    cluster_window=plot_clusters(clusters,A.shape)
    pts,flikapts_fname=getPoints(clusters,A,image_location)
    
    original.setAsCurrentWindow()
    load_points(flikapts_fname)
    background(cluster_window,original,.2,True)
    close(cluster_window)
    
    toc=time.time()-tic
    print('Total Running Time = {} s'.format(toc))
###############################################################################
###############################################################################








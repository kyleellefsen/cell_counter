# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:58:53 2015

@author: Kyle Ellefsen
"""
import os, sys
import numpy as np
import matplotlib
from skimage.measure import regionprops
from skimage.measure import label
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from flika import global_vars as g
from flika.window import Window
from flika.roi import ROI_rectangle, makeROI
from flika.process.file_ import open_file
from flika.utils.misc import open_file_gui
from flika.process import *
from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox
from .get_densities import getDensities_multi #  from plugins.cell_counter.get_densities import getDensities_multi
from .get_higher_points import getHigherPoints_multi # from plugins.cell_counter.get_higher_points import getHigherPoints_multi



    
def normalize(A):
    A = np.copy(A)
    std = np.std(A)
    mean = np.mean(A)
    A -= mean
    A /= std
    return A
    
class Point():
    def __init__(self,idx):
        self.children = []
        self.idx = idx

    def __repr__(self):
        return str(self.idx)

    def getDescendants(self):
        self.descendants = self.children[:]
        for child in self.children:
            self.descendants.extend(child.getDescendants())
        return self.descendants




def plot_higher_pts(higher_pts):
    higher_pts_tmp=higher_pts[higher_pts[:,0]>1]
    y=[d[0] for d in higher_pts_tmp] #smallest distance to higher point
    x=[d[2] for d in higher_pts_tmp] # density 
    pw=pg.PlotWidget()
    pw.plot(x,y,pen=None, symbolBrush=QtGui.QBrush(QtCore.Qt.blue), symbol='o')
    pw.plotItem.axes['left']['item'].setLabel('Smallest distance to denser point'); pw.plotItem.axes['bottom']['item'].setLabel('Density')
    pw.show()
    return pw

def find_clusters(higher_pts,idxs,center_minDensity,center_minDistance):
    centers = []
    outsideROI = []
    for i in np.arange(len(higher_pts)):
        y = higher_pts[i][0]  # smallest distance to higher point
        x = higher_pts[i][2]  # density
        if x > center_minDensity and y > center_minDistance:
            centers.append(i)
        else:
            outsideROI.append(i)

    higher_pts2 = higher_pts[:,1].astype(np.int)  # contains index of next highest point
    points = [Point(i) for i in np.arange(len(higher_pts2))]
    loop = np.arange(len(higher_pts2))
    loop = np.delete(loop, centers)
    for i in loop:
        if higher_pts2[i] != i:
            points[higher_pts2[i]].children.append(points[i])
    clusters = []
    for center in centers:
        descendants = points[center].getDescendants()
        cluster=[d.idx for d in descendants]
        cluster=np.array(cluster+[center])
        clusters.append(cluster)
        
    clusters_idx = clusters
    clusters = [idxs[c] for c in clusters]
    return clusters, clusters_idx
    

def plot_clusters(clusters,imshape):
    mx, my = imshape
    cluster_im = np.zeros((mx,my,4))
    cmap = matplotlib.cm.gist_rainbow
    for i in np.arange(len(clusters)):
        color = cmap(int(((i%15)*255./16)))#+np.random.randint(255./12)))
        cluster = clusters[i]
        x,y = cluster.T
        cluster_im[x,y,:]=color
    return Window(cluster_im, 'Clusters')
    
def plot_cluster2(clusters):
    sizes = [np.max(cluster,0)-np.min(cluster,0) for cluster in clusters]
    np.max(sizes)
    mmx, mmy = np.max(sizes,0)+1
    Cells = np.zeros((len(clusters), mmx, mmy))
    for i, cluster in enumerate(clusters):
        x,y = (cluster-np.min(cluster,0)).T
        Cells[i,x,y] = 1
    W = Window(Cells)
    return W
    
    
def filter_bad_cells(clusters, min_number_of_pixels_in_cell):
    new_clusters=[]
    for i, cluster in enumerate(clusters):
        if len(cluster) < min_number_of_pixels_in_cell:
            continue
        offset = np.min(cluster,0)
        cluster = cluster-offset
        mmx, mmy = np.max(cluster, 0) + 1
        image = np.zeros((mmx, mmy), dtype=np.uint8)
        x, y = cluster.T
        image[x, y] = 1
        label_image = label(image, background=0)
        regions = regionprops(label_image)
        if len(regions) == 1:
            new_clusters.append(cluster+offset)
        else:
            biggest_idx = np.argmax([region.area for region in regions])
            biggest = regionprops(label_image)[biggest_idx]
            if biggest.area > min_number_of_pixels_in_cell:
                new_clusters.append(biggest.coords+offset)
    return new_clusters


def getprops(clusters, image):
    properties = []
    for i, cluster in enumerate(clusters):
        offset = np.min(cluster,0)
        cluster = cluster-offset
        mmx,mmy = np.max(cluster,0)+1
        im = np.zeros((mmx,mmy),dtype=np.uint8)
        x,y = cluster.T
        im[x,y] = 1
        label_image = label(im,background=0)
        label_image += 1
        regions = regionprops(label_image)
        if len(regions) > 1:
            print("ERROR You should have run this through 'filter_bad_cells' first" )
            return None
        r = regions[0]
        x = x + offset[0]
        y = y + offset[1]
        mean_vals = np.mean(image[x,y])
        meanx = np.mean(x)
        meany = np.mean(y)
        properties.append([i, meanx, meany, r.area, r.eccentricity, r.orientation, mean_vals, ])
    properties = np.array(properties)
    return properties


def getPoints(clusters, Image, image_location, saveFlikaPoints=True):
    mean_xs = []; mean_ys = []; mean_vals = []
    for cluster in clusters:
        x,y = cluster.T
        vals = Image[x,y]
        mean_xs.append(np.sum(vals*x)/np.sum(vals))
        mean_ys.append(np.sum(vals*y)/np.sum(vals))
        mean_vals.append(np.mean(vals))
        
    pts = np.array([mean_xs, mean_ys, mean_vals]).T
    #pts=pts[pts[:,2]>400]
    if saveFlikaPoints:
        filename=os.path.splitext(image_location)[0]+'.txt'
        flika_pts=np.hstack((np.zeros((len(pts),1)),pts[:,:2]))
        np.savetxt(filename, flika_pts)
        return pts, filename
    else:
        return pts, None


class Cell_Counter():
    """cell_counter()
    """

    def __init__(self):
        pass

    def gui(self):
        pass

cell_counter = Cell_Counter()
    
def launch_docs():
    url='https://github.com/kyleellefsen/cell_counter'
    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

def run_demo():
    """
    For testing, run the following lines:
    
    from plugins.cell_counter.cell_counter import *
    test_file = r"C:/Users/kyle/.FLIKA/plugins/cell_counter/test/1b_01_cropped.tif"
    
"""

    ###############################################################################
    #                    CONSTANTS
    ###############################################################################
    mask_radius = 19
    thresh = .5
    density_thresh = 50
    center_minDensity = 0
    center_minDistance = 8
    gaussianblur_sigma = 20
    min_number_of_pixels_in_cell = 50
    test_file = os.path.join(os.path.dirname(__file__), 'test', '1b_01_cropped.tif')  # test_file = r"C:\Users\kyle\.FLIKA\plugins\cell_counter\test\1b_01_cropped.tif"
    ###############################################################################
    ###############################################################################

    original = open_file(test_file)
    blurred = gaussian_blur(gaussianblur_sigma, norm_edges=False, keepSourceWindow=True)
    high_pass = image_calculator(original, blurred, 'Subtract', keepSourceWindow=True)
    close(blurred)
    A_norm = normalize(high_pass.image)
    close(high_pass)
    Densities = getDensities_multi(A_norm, thresh, mask_radius)
    Window(Densities, 'Densities')
    higher_pts, idxs = getHigherPoints_multi(Densities, density_thresh)

    # pw=plot_higher_pts(higher_pts)
    clusters, clusters_idx = find_clusters(higher_pts, idxs, center_minDensity, center_minDistance)
    # plot_clusters(clusters, Densities.shape)
    clusters = filter_bad_cells(clusters, min_number_of_pixels_in_cell)
    cluster_window = plot_clusters(clusters, Densities.shape)
    pts, flikapts_fname = getPoints(clusters, A_norm, test_file)
    original.setAsCurrentWindow()
    open_points(flikapts_fname)
    background(cluster_window, original, .2, True)
    close(cluster_window)

    ###############################################################################
    ###############################################################################






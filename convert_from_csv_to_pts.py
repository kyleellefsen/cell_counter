# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:55:47 2015

@author: Kyle Ellefsen
"""

import numpy as np
import os

filename='D:/Desktop/1b_20.csv'
filename_o=os.path.splitext(filename)[0]+'.txt'
csv = np.genfromtxt (filename, delimiter=",")
flika_pts=np.hstack((np.zeros((len(csv),1)),csv[:,2:]))
np.savetxt(filename_o,flika_pts)
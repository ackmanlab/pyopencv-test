---
date: 2016-06-20 16:22:23  
tags: analysis, python, wholeBrain, data visualization  
author: James B. Ackman  
---

# python opencv testing

This is a python/opencv analysis repo to test:  

* read capability of hdf5 matlab d2r.mat files with h5py and hemispheric symmetry calculations using numpy/opencv
* read capability of tiff movies and make dF/F (F-F0/F0) normalization and display using numpy/opencv

The python notebooks have contain some subfunctions and routines that are potentially instructive. The images saved in assets are a number of snapshot frame of bi-hemispheric activity patterns that were saved by pressing `s` while playing movies in `2016-06-15-opencv-RGBmasks-xcorr.ipynb`

See also [anaconda python installation and opencv setup instructions](http://ackmanlab.com/2016/06/20/install-ipython.html)

For companion functions based on some of the scripts implemented in these notebooks see `hemisphereXcorr.py` and `playtiffmovie.py` at [pyWholeBrain](https://github.com/ackmanlab/pyWholeBrain).

---
date: 2016-06-20 16:22:23  
tags: analysis, python, wholeBrain, data visualization  
author: James B. Ackman  
---

# pyopencv-test

This is a python/opencv analysis repo to test:  

* test read capability of hdf5 matlab d2r.mat files with h5py and hemisphere symmetry calculations using numpy/opencv
* test read capability of tiff movies and make dF/F (F-F0/F0) normalization and display using numpy/opencv

See also [anaconda python installation and opencv setup instructions](http://ackmanlab.com/2016/06/20/install-ipython.html)

Standalone functions (details to follow):

`hemisphereXcorr.py` : do xcorr calculations on a cluster of amazon EC2 instances

`playtiffmovie.py`: run this from command line (on osx or windows or linux) to open up 1 or more tiff files, concatenate, and display as dF/F in opencv window. Hopefully we can use this to replace the [ImageJ macro dFoFmovie](https://gist.github.com/ackman678/11155761) on local or acquisition pcs.



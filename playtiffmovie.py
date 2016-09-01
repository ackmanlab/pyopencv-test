#!/usr/bin/env python
#playtiffmovie.py
#ipython -i -c "%run playtiffmovie.py 'file1.tif'
#python playtiffmovie.py 'file1.tif' 'file2.tif'
#python playtiffmovie.py '151103_05.tif' $(ls 151103_05@*.tif)
#James B. Ackman 2016-06-20 16:47:02  

import os, sys
from timeit import default_timer as timer
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import h5py
import tifffile
#from joblib import Parallel, delayed
# %matplotlib inline
print("OpenCV Version : {0}".format(cv2.__version__))
print("Numpy Version : {0}".format(np.__version__))


def playMovie(A,newMinMax=False):
    #play movie in opencv after normalizing display range
    #A is float input (float32)
    #newMinMax is an optional tuple of length 2, the new display range
    #Note: the array normalization is done inplace, thus the array will be rescaled outside scope of this function (but will still be float32)
    cv2.startWindowThread()
    cv2.namedWindow("raw", cv2.WINDOW_NORMAL) #Create a resizable window
    i = 0
    
    #Normalize movie range and change to uint8 before display
    t0 = timer()
    sz = A.shape
    A = np.reshape(A, (sz[0], A.size/sz[0]))
    meanA,stdA = cv2.meanStdDev(A)
    print("mean: {0}, std: {1}".format(meanA,stdA))
    
    if newMinMax == False:
        newMin = meanA - 3*stdA
        newMax = meanA + 7*stdA
    else:
        newMin = newMinMax[0]
        newMax = newMinMax[1]
    
    newSlope = 255.0/(newMax-newMin)
    #A += abs(np.amin(A))
    cv2.subtract(A, newMin, A)
    cv2.multiply(A, newSlope, A)
    A = np.reshape(A, sz)
    A = A.astype('uint8', copy=False)
    print("Movie range normalization: {0}".format(timer()-t0))

    toggleNext = True
    tf = True
    while True:
        #im = np.uint8(A3[:,:,i] * 255)
        im = A[i,:,:]
        #im = cv2.GaussianBlur(im,(0,0),3)
        #th,bw = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #bw = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,0)
        im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
        cv2.putText(im, str(i), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255)) #draw frame text
        cv2.imshow('raw',im)
        k = cv2.waitKey(10) 
        if k == 27: #if esc is pressed
            break
        elif (k == ord(' ')) and (toggleNext == True): #if space is pressed
            tf = False
        elif (k == ord(' ')) and (toggleNext == False): #if space is pressed
            tf = True
        toggleNext = tf #toggle the switch
        if k == ord('b') and toggleNext:
            i -= 100
        elif k == ord('f') and toggleNext:
            i += 100
        elif k == ord('m') and (toggleNext == False):
            i += 1
        elif k == ord('n') and (toggleNext == False):
            i -= 1
        elif toggleNext:
            i += 1
        
        if (i > (A.shape[0]-1)) or (i < 0) :
            i = 0

    cv2.destroyAllWindows()


def main():
    try:
        fn = [sys.argv[1]]
    except:
        print("one movie file required")
    
    nfiles = len(sys.argv)-1 #first sys.argv is the script name
    print("No. of input files: {0}".format(nfiles))
    if nfiles > 1:
        for i in np.arange(2,nfiles+1):
            fn.append(sys.argv[i])
    
    t0 = timer()
    # fn = '/waves/videos/140509/140509_22.tif'
    #A = tifffile.imread(fn)
    with tifffile.TiffFile(fn[0]) as tif:
         A = tif.asarray()
    print("Load movie: {0} sec".format(timer()-t0))
    
    if nfiles > 1:
        #reshape multiple arrays into one
        # A = np.reshape(A, (A.shape[0]*A.shape[1], A.shape[2], A.shape[3]))
        Alist = [A]
        for i in np.arange(1,nfiles):
            t0 = timer()
            with tifffile.TiffFile(fn[i]) as tif:
                 Alist.append(tif.asarray())
            print("Load movie: {0} sec".format(timer()-t0))
        
        t0 = timer()
        #Alist = tuple(Alist)
        A2 = np.concatenate(Alist, axis=0)
        print("numpy cat arrays {0} sec".format(timer()-t0))
        del(Alist)
    else:
        A2 = A
    del(A)
    
    sz = A2.shape
    A2 = np.reshape(A2, (sz[0], A2.size/sz[0]))
    print(A2.shape)

    t0 = timer()
    Amean = np.mean(A2,axis=0,dtype='float32')
    #A2 = np.reshape(A2, (A2.shape[0], A.shape[2], A.shape[3]))
    #Amean = cv2.reduce(A2,0,cv2.REDUCE_AVG)
    #Amean = np.add.reduce(A2, 0)
    #Amean /= A2.shape[0]
    print("z mean: {0} sec".format(timer()-t0))
    print("Amean shape: {0}".format(Amean.shape))
    #Amean = np.reshape(Amean, (sz[1],sz[2]))
    #A2 = np.reshape(A2, sz)
    print("Amean type: {0}".format(Amean.dtype))

    t0 = timer()
    A2 = A2.astype('float32', copy=False)
    print("float32: {0} sec".format(timer()-t0))

    t0 = timer()
    # for i in np.arange(A2.shape[0]):
    #     A2[i,:,:] = ((A2[i,:,:] / Amean) - 1.0)
    for i in np.arange(A2.shape[0]):
        A2[i,:] /= Amean
        A2[i,:] -= 1.0

    print("dfof normalization: {0} sec".format(timer()-t0))
    A2 = np.reshape(A2, sz)
    print(A2.dtype)
    print(A2.shape)
    playMovie(A2)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
#hemisphereXcorr.py
#Compute xcorr metric between hemispheres. Reads in v7.3 matlab files (hdf5 format) saved as d2r.mat from wholeBrainDX. Runs as a batch process, requires a space delimited txt file of filenames as input.
#TODO: edit so that an alternate raw image or detected signal mask array can be passed as input.
#Usage: hemisphereXcorr.py 'filesAll.txt 'dXcorrObs.txt'
#ipython -i -c "%run hemisphereXcorr.py 'filesAll.txt' 'dXcorrObs.txt'"
#python hemisphereXcorrParallel.py 'filesS3.txt' 'dXcorrObs.txt' '30' 'threading' '2'
#James B. Ackman 2016-06-01 13:52:33  
import pandas as pd
import os, sys
import datetime
import subprocess
import cv2
#%matplotlib inline
import numpy as np
#import matplotlib.pyplot as plt
#import scipy.io as spio
import h5py
from skimage.draw import polygon
#from skimage import data
#from scipy import ndimage as ndi
#from scipy import signal as signal
#from skimage.util import random_noise
#from skimage.feature import match_template
from timeit import default_timer as timer
from joblib import Parallel, delayed


def importFilelist(filepath):
    #import text file of filenames as pandas dataframe
    files = pd.read_table(filepath,names=['filename','matfilename', 'age.g'],sep=' ',header=None)
    return files


def writeData(filepath,data):
    #write results dataframe by appending to file
    #filepath = 'dXcorrObs.txt'
    if os.path.isfile(filepath):
        writeHeader=False
    else:
        writeHeader=True

    with open(filepath, 'a') as f:
        data.to_csv(f, index=False, sep='\t', header=writeHeader)  #make if logic to not append with header if file already exists


def getA3binaryArray(f,region):
    #read in domainData into A3 binary array
    sz = np.int64(region['domainData']['CC/ImageSize'][:,0])
    A3 = np.zeros(sz,dtype=bool)
    for i in np.arange(region['domainData']['CC/PixelIdxList'].shape[0]):
        pxind = np.array(f[region['domainData']['CC/PixelIdxList'][i,0]], dtype='int64')
        A3.T.flat[pxind[0,:]] = True
    return A3


def getHemisphereMasks(f,region,A3):
    #read region['coords'] and region['name'] to get hemisphere masks
    sz = A3.shape
    leftMask = np.zeros((sz[0],sz[1]),dtype=bool)
    rightMask = np.zeros((sz[0],sz[1]),dtype=bool)
    bothMasks = np.zeros((sz[0],sz[1]),dtype=bool)
    for i in np.arange(region['name'].len()):  #this is length 33
        currentString = f[region['name'][i,0]][...].flatten().tostring().decode('UTF-16')
        if (currentString == 'cortex.L') or (currentString == 'cortex.R'):
            #print(i,currentString)
            x = np.array(f[region['coords'][i,0]],dtype='int64')[0,:]
            y = np.array(f[region['coords'][i,0]],dtype='int64')[1,:]
            x = np.append(x,x[0]) #close the polygon coords
            y = np.append(y,y[0]) #close the polygon coords
            rr, cc = polygon(y,x,bothMasks.shape)
            if currentString == 'cortex.L':
                x_L = x
                y_L = y
                leftMask[rr,cc] = True
            elif currentString == 'cortex.R':
                x_R = x
                y_R = y
                rightMask[rr,cc] = True
            bothMasks[rr,cc] = True
    return leftMask, rightMask


def stridemask3d(mask,z):
    #make a 2D image mask into a 3D repeated image mask using numpy stride_tricks to fake the 3rd dimension
    mask3d = np.lib.stride_tricks.as_strided(
                    mask,                              # input array
                    (mask.shape[0], mask.shape[1],z),  # output dimensions
                    (mask.strides[0], mask.strides[1],0) # stride length in bytes
                )
    return mask3d


def cvblur(image,template):
    #fast gaussian blur with cv2
    image = cv2.GaussianBlur(image,(0,0),3)
    template = cv2.GaussianBlur(template,(0,0),3)
    return image, template


def cvcorr(image,template):
    #faster xcorr with cv2
    c = cv2.matchTemplate(image,template,cv2.TM_CCORR_NORMED)
    corrValue = c[c.shape[0]/2,c.shape[1]/2]
    return corrValue


def getA3hemipshereArrays(A3,leftMask,rightMask):
    #calculate lateral column shift for a flipped right hemisphere for a common coord system
    rightMask_flip = np.fliplr(rightMask)
    IND_L = np.nonzero(leftMask) #dim2 of the tuple is the col indices
    IND_Rflip = np.nonzero(rightMask_flip) #dim2 of the tuple is the col indices
    dx = np.amax(IND_L[1]) - np.amax(IND_Rflip[1]) #number of pixesl to shift the flipped right hemisphere column indices by

    #get 3d mask arrays, new binary data arrays, and a flipped R hemi array
    leftMask3d = stridemask3d(leftMask,A3.shape[2])
    rightMask3d = stridemask3d(rightMask,A3.shape[2])

    A3left = np.logical_and(A3, leftMask3d)
    A3right = np.logical_and(A3, rightMask3d)
    A3right_flip = np.fliplr(A3right)
    
    #make a new shifted right hemisphere array
    IND_R3d = np.nonzero(A3right_flip) #dim2 of the tuple is the col indices
    INDnew = IND_R3d[1] + dx #add shift to column indices
    A3right_shift = np.zeros(A3.shape,dtype=bool)
    A3right_shift[IND_R3d[0],INDnew,IND_R3d[2]] = True
    return A3left, A3right_shift


def getCorr(A3left,A3right,i):
    image = A3left[:,:,i].astype('uint8') * 255
    template = A3right[:,:,i].astype('uint8') * 255 #cv2.matchTemplate needs uint8
    image,template = cvblur(image,template)
    obsCorr = cvcorr(image,template)
    fr = i + 1
    return fr, obsCorr


def getCorrRand(A3left,A3right,i,j):
    image = A3left[:,:,i].astype('uint8') * 255
    template = A3right[:,:,i].astype('uint8') * 255 #cv2.matchTemplate needs uint8
    image,template = cvblur(image,template)
    obsCorr = cvcorr(image,template)
    fr = i + 1
    return fr, j, obsCorr


def fetchResults(filename,matfilename,njobs=2,backendType='threading',verbose_num=10,nshuffle=100,maxbytes=None):
    #read in region matlab v7.3 hdf5 file to python
    #matfile = '../150123_06_20160418-233029_d2r.mat'
    process = subprocess.Popen(["aws", "s3", "cp", "s3://data.ackmanlab.com/"+matfilename, "/home/ubuntu/data/"+matfilename, "--region", "us-west-2"])
    (output, err) = process.communicate()
    f = h5py.File(matfilename,'r') 
    region = f.get('region')
    A3 = getA3binaryArray(f,region)
    leftMask, rightMask = getHemisphereMasks(f,region,A3)
    A3left, A3right_shift = getA3hemipshereArrays(A3,leftMask,rightMask)
    # frameArray = np.zeros(A3.shape[2],dtype=np.int)
    # autoArray = np.zeros(A3.shape[2],dtype=np.int)
    # obsArray = np.zeros(A3.shape[2])
    #loop through current dataset loaded in current workspace
    t0 = timer()
    resultsObs = Parallel(n_jobs=njobs, backend=backendType, verbose=verbose_num, max_nbytes=maxbytes)(delayed(getCorr)(A3left, A3right_shift, i) for i in np.arange(A3left.shape[2]))
    # if (i % 500) == 1:
    #     print("fr {0}/{1}".format(i,A3left.shape[2]))
    print("Elapsed time obs: {0}".format(timer() - t0))
    sys.stdout.flush()
    
    t0 = timer()
    for j in np.arange(nshuffle):
        ind = np.arange(A3right_shift.shape[2])
        np.random.shuffle(ind)
        A3right_shuffle = A3right_shift[:,:,ind]
        results = Parallel(n_jobs=njobs, backend=backendType, verbose=verbose_num, max_nbytes=maxbytes)(delayed(getCorrRand)(A3left, A3right_shuffle, i, j) for i in np.arange(A3left.shape[2]))
        if j < 1:
            resultsRand = results
        else:
            resultsRand.extend(results)

        if (j % 50) == 1:
            print("shuffle iteration {0}/{1}".format(j,nshuffle))
            sys.stdout.flush()
    
    print("Elapsed time random: {0}".format(timer() - t0))
    sys.stdout.flush()

    dfObs = pd.DataFrame(resultsObs,columns=['fr','obsCorr'])
    dfObs['filename'] = filename
    dfRand = pd.DataFrame(resultsRand,columns=['fr','iteration','randCorr'])
    dfRand['filename'] = filename
    return dfObs, dfRand



def main():
    import sys
    #filepath = '/Users/ackman/Dropbox/notes/wholeBrain_analysis/quantif_20160205/filesAll.txt'
    try:
        filepath = sys.argv[1]
    except:
        print("space delimited filelist (with full path in second column) required")
    
    try:
        outFile = sys.argv[2]
    except:
        outFile = 'dXcorr.txt'
    
    try:
        njobs = np.int(sys.argv[3])
    except:
        njobs = 1
    
    try:
        backendType = sys.argv[4]
    except:
        backendType = 'threading'

    try:
        verbose_num=np.int(sys.argv[5])
    except:
        verbose_num=0
    
    try:
        amiIndex=np.int(sys.argv[6])
    except:
        amiIndex=0

    try:
        nNodes=np.int(sys.argv[7])
    except:
        nNodes=1

    try:
        nshuffle = np.int(sys.argv[8])
    except:
        nshuffle = 1

    files = importFilelist(filepath)
    print("start: {0}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
    sys.stdout.flush()
    
    #make indexing of filelist based on current machine instance id and total no. of compute nodes being used
    nFiles = files.shape[0]
    nper = nFiles/nNodes
    nrem = nFiles % nNodes
    
    if amiIndex <= (nrem-1):
        idx=np.append(np.arange(nper*amiIndex,nper*amiIndex+nper), nper*nNodes+amiIndex)
    else:
        idx=np.arange(nper*amiIndex,nper*amiIndex+nper)
    
    #start the batch processing
    print('parallel AWS!')
    sys.stdout.flush()
    i = amiIndex
    for i in idx:
        outFile1 = outFile[:-4] + 'Obs' + str(amiIndex) + '.txt'
        outFile2 = outFile[:-4] + 'Rand' + str(amiIndex) + '.txt'
        print("\n---processing {0}---".format(files['filename'][i]))
        sys.stdout.flush()
        dfObs,dfRand = fetchResults(files['filename'][i],files['matfilename'][i],njobs,backendType,verbose_num,nshuffle)
        writeData(outFile1,dfObs)
        writeData(outFile2,dfRand)

    print("finish: {0}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
    sys.stdout.flush()
    


if __name__ == '__main__':
    main()

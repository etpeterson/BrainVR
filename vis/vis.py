'''
This program is intended to visualize data and match it up with a webcam image

opengl
http://www.siafoo.net/snippet/142
http://www.siafoo.net/snippet/310
maybe best to just use vtk
http://wiki.scipy.org/Cookbook/vtkVolumeRendering?action=show&redirect=vtkVolumeRendering

Created on Jan 27, 2015

@author: Eric Peterson
'''


#import numpy as np
#import OpenGL
import sys
import cv2

waitkey=0
vid=0


if len(sys.argv)<2:
    print 'No input argument used, attempting to open the camera'
elif len(sys.argv)==2:
    print 'Attempting to open the video ',str(sys.argv[1])
    vid=str(sys.argv[1])
else:
    print 'I do not understand the input arguments'
    print 'It should be ',str(sys.argv[0]),' <movie file name'

#load a video
cap=cv2.VideoCapture(vid) #sys.argv[1]

while(cap.isOpened() and waitkey!=ord('q')):
    #capture frame
    ret, frame = cap.read()
    
    #display the movie
    cv2.imshow('frame',frame)
    waitkey=cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
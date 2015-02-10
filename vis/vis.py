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
import numpy as np


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = img1#np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = img2#np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    #return the image
    return out

    # Show the image
    #cv2.imshow('Matched Features', out)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
    

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
surf=cv2.SURF(1900) #create a SURF object
match=cv2.BFMatcher() #create a matcher

while(cap.isOpened() and waitkey!=ord('q')):
    #capture frame
    (ret, frame) = cap.read()
    #frame=cv2.cvtColor(frame,cv2.CV_RGB2GRAY)
    
    #SURF processing
    (kp, des) = surf.detectAndCompute(frame,None)
    #kp=surf.detect(frame,None)
    #print len(kp)
    #print kp[0]
    #kp.
    
    if waitkey==ord('s') or 'frame_static' not in locals():
        #print "saving image"
        frame_static=frame.copy()
        (kp_static,des_static)=surf.detectAndCompute(frame_static,None)
    #else:
    #    frame_static=frame.copy()
    #    kp_static=list(kp)
    #    des_static=list(des)
    
    matches=match.match(des,des_static)
    
    #frame_kp = cv2.drawKeypoints(frame,kp,None,(255,0,0),4)
    
    #frame_kp=cv2.drawMatches(frame,kp,frame_static,kp_static,matches,None)
    matches=drawMatches(frame,kp,frame_static,kp_static,matches)
    
    #display the movie
    cv2.imshow('frame',matches)
    waitkey=cv2.waitKey(1)
    
    
    


cap.release()
cv2.destroyAllWindows()

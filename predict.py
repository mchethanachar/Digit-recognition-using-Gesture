
 '''
Written by:Chethan M
           BMS College of Engineering
           Bangalore
           mchethan.achar@gmail.com
Programming languge: Python 3.6.8
Last modified:20th jan 2019
'''
# importing OpenCV, time and Pandas library
import numpy as np
import imutils
import cv2, time, pandas
# importing datetime class from datetime library
from datetime import datetime
from PIL import Image
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model('C:/Users/india/learnt_models/mnistCNN.h5')


index=1
dim=(28,28)
# Assigning our static_back to None
static_back = None
#A plain white image of png format is used
MyScreen=cv2.imread("white.png")
# List when any moving object appear
motion_list = [ None, None ]

pX=-1
pY=-1

# Time of movement
time = []
 
# Initializing DataFrame, one column is start 
# time and other column is end time
df = pandas.DataFrame(columns = ["Start", "End"])
 
# Capturing video
video = cv2.VideoCapture(0)

# Infinite while loop to treat stack of image as video
while True:
    # Reading frame(image) from video
    check, frame = video.read()
    motion=0;
    cv2.imwrite("abc2.jpg",frame)
    image = cv2.imread("abc2.jpg")
    if MyScreen is None:
            MyScreen=cv2.imread("white.png")
    #nCV and Python Color DetectionPython

    # define the list of boundaries
    lower=[17, 15, 100]
    upper=[50, 56, 200]

    #blue
    #lower=[102,35,0]
    #upper=[255,204,204]
    
    #lower=[255, 255, 255]
    #upper=[255, 255, 255]
    
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
 
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    
    # Displaying image in gray_scale
    cv2.imwrite("abc3.jpg",output)
    
    image = cv2.imread("abc3.jpg")
    image=cv2.flip(image,1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)[1]


    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	    cv2.CHAIN_APPROX_SIMPLE)
    
    
    #print(imutils.is_cv2())

    
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts=cnts[0]

   
    l=len(cnts)
    max=0
    i=0
    ind=0
    while i < l:
        area = cv2.contourArea(cnts[i])
        if max<area:
            max=area
            ind=i
        i=i+1
    if l>0:
        
        M = cv2.moments(cnts[ind])
        if M["m00"]==0:
            cX=0
            cY=0
        else:
            
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        
    # draw the contour and center of the shape on the image
        cv2.drawContours(image, [cnts[ind]], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
        cv2.circle(MyScreen, (cX+200, cY),22 , (0, 0, 0), -1)
        if pX!=-1 and pY!=-1:
            cv2.line(MyScreen, (pX, pY),(cX+200, cY), (0,0, 0),22)
        pX=cX+200
        pY=cY
        cv2.putText(image, "center", (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
    # show the image
    
    #cv2.imshow("Image", image)
 
    cv2.imshow("Name", MyScreen)
 
    key = cv2.waitKey(1)
    
    if key == ord(' '):
        MyScreen=None
        pX=-1
        pY=-1
    if key == ord('z'):
        img = cv2.resize(255-MyScreen, dim)
        cv2.imwrite('D:/ML_data/final.png',img)
        img = Image.open('D:/ML_data/final.png').convert("L")
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1).astype('float32')
        im2arr=im2arr/255
        y_pred = model.predict_classes(im2arr)
        print("The predicted number is   ",end='')
        print(y_pred)
        MyScreen=None
        pX=-1
        pY=-1



        
        
    if key == ord('h'):
        #resized = cv2.resize(MyScreen, dim, interpolation = cv2.INTER_AREA)
        resized = cv2.resize(255-MyScreen, dim)
        cv2.imwrite('D:/ML_data/mnist_test_'+str(index)+'.png',resized)
        print(MyScreen.shape)
        print(index)
        index=index+1
        MyScreen=None
        pX=-1
        pY=-1
    # if q entered whole process will stop
    if key == ord('q'):
        if motion == 1:
            time.append(datetime.now())
        # if something is movingthen it append the end time of movement
        break
 
# Appending time of motion in DataFrame


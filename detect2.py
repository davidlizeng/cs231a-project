import cv2
import cv
import numpy as np

def detectLetters(img, size):
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    img_lap = cv2.Laplacian(img_gray,cv2.CV_8U)
    #cv2.imwrite('lap.png', img_lap)
    #img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 0, 1, ksize=5)
    #cv2.imwrite('sobel.png', img_sobel)
    #img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, 3, 1, 0, cv2.BORDER_DEFAULT)
    unused, img_threshold = cv2.threshold(img_lap, 0, 255, cv.CV_THRESH_OTSU + cv.CV_THRESH_BINARY)
    #cv2.imwrite('thresh.png', img_threshold)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, size )
    morphed = cv2.morphologyEx(img_threshold, cv.CV_MOP_CLOSE, element)
    cv2.imwrite('morph.png', morphed)
    contours, unused = cv2.findContours(morphed, 0, 1)
    boundRect = []
    for i in xrange(len(contours)):
        if len(contours[i]) > 10:
            contour_poly = cv2.approxPolyDP(contours[i], 3, True)
            appRect = cv2.boundingRect(contour_poly)
            boundRect.append(appRect)
    return boundRect

def detectChars(img, size):
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray,(5,5),0)
    img_lap = cv2.Laplacian(img_gray,cv2.CV_8U)
    img_sobel = cv2.Sobel(img_gray, cv2.CV_8U,1,0,ksize=3)
    unused, img_threshold = cv2.threshold(img_lap, 0, 255, cv.CV_THRESH_OTSU + cv.CV_THRESH_BINARY)
    #img_threshold = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    cv2.imwrite('thresh.png', img_threshold)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, size )
    morphed = cv2.morphologyEx(img_threshold, cv.CV_MOP_CLOSE, element)
    cv2.imwrite('morph.png', morphed)
    #contours, unused = cv2.findContours(morphed, 0, 1)
    contours,hierarchy = cv2.findContours(morphed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    boundRect = []
    for i in xrange(len(contours)):
        contour_poly = cv2.approxPolyDP(contours[i], 3, True)
        appRect = cv2.boundingRect(contour_poly)
        boundRect.append(appRect)
    return boundRect

img = cv2.imread('doc1.png')
img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
boundRect = detectLetters(img, (12,12))
for rect in boundRect:
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1] + rect[3]), (0,255,0))
    box = np.array(img_gray[rect[1]-1:rect[1]+rect[3]+1,rect[0]-1:rect[0]+rect[2]+1])
cv2.imwrite('doc1-box.png', box)
cv2.imwrite('doc1-morph.png', img)
boximg = cv2.imread('doc1-box.png')
boundRect = detectChars(boximg, (2,2))
for rect in boundRect:
    cv2.rectangle(boximg, (rect[0]-1, rect[1]-1), (rect[0]+rect[2], rect[1] + rect[3]), (0,255,0))
cv2.imwrite('doc1-box-morph.png', boximg)


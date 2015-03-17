import cv
import cv2
import numpy as np
import sys
import itl_paragraph

IMAGE_FILE = 'images/rotate1.png'
[IMAGE_NAME, EXTENSION] = IMAGE_FILE.split('.')

def parsePaper(img):
    img_sob = cv2.Sobel(img,cv2.CV_8U,1,1,ksize=7)
    img_lap = cv2.Laplacian(img, cv2.CV_8U)
    unused, img_threshold = cv2.threshold(img_sob, 0, 255, cv.CV_THRESH_OTSU + cv.CV_THRESH_BINARY)

    # Blur to get paragraph blocks
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    morphed = cv2.morphologyEx(img_threshold, cv.CV_MOP_CLOSE, element)

    # Use RETR_EXTERNAL to remove boxes that are completely contained by the paragraph
    contours, hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundRects = []
    for i in xrange(len(contours)):
        contourPoly = cv2.approxPolyDP(contours[i], 3, True)
        boundRect = cv2.boundingRect(contourPoly)
        boundRects.append(boundRect)

    paperWidth = img.shape[1]
    equationRects = []
    textRects = []
    for rect in boundRects:
        if rect[2]*rect[3] > 250:
            if (rect[0]*1.0)/paperWidth > 0.225:
                equationRects.append(rect)
            else:
                textRects.append(rect)

    # Extract lines from paragraph
    paragraphs = []
    for rect in boundRects:
        [x, y, w, h] = rect
        paragraph = img[y:(y+h), x:(x+w)]
        paragraphs.append(paragraph)

    maxWidth = 0
    for rect in boundRects:
        if maxWidth < rect[2]:
            maxWidth = rect[2]

    return maxWidth

def genRotateImage(img, r, center):
    rotMat = cv2.getRotationMatrix2D(center, r, 1)
    rotMat[0][2] += img.shape[1]/2
    rotMat[1][2] += img.shape[0]/2
    img_invert = (255 - img)
    img_r = cv2.warpAffine(img_invert, rotMat, (2*img.shape[1], 2*img.shape[0]))
    img_r = (255 - img_r)
    return img_r

def findBestRotation(img_input):
    img = cv2.cvtColor(img_input, cv.CV_BGR2GRAY)
    center = (img.shape[0]/2, img.shape[1]/2)
    r = 0
    step = 0.6
    img_r = genRotateImage(img, r, center)
    bestWidth = parsePaper(img_r)
    negr = r - step
    img_r = genRotateImage(img, negr, center)
    negWidth = parsePaper(img_r)
    posr = r + step
    img_r = genRotateImage(img, posr, center)
    posWidth = parsePaper(img_r)
    direction = 0
    if negWidth < bestWidth and negWidth < posWidth:
        direction = -1
    elif posWidth < bestWidth and posWidth < negWidth:
        direction = 1
    if direction != 0:
        nextWidth = bestWidth
        while bestWidth >= nextWidth:
            bestWidth = nextWidth
            r = r + step*direction
            img_r = genRotateImage(img, r, center)
            nextWidth = parsePaper(img_r)
        r = r - step*direction

    rotMat = cv2.getRotationMatrix2D(center, r, 1)
    img_invert = (255 - img)
    img_r = cv2.warpAffine(img_invert, rotMat, (img.shape[1], img.shape[0]))
    img_r = (255 - img_r)
    if DEBUG:
        print r
        cv2.imwrite(IMAGE_NAME + '-rotated.' + EXTENSION, img_r)
        cv2.imshow('Best Rotation', img_r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return r

def test():
    global DEBUG
    DEBUG = True
    img = cv2.imread(IMAGE_FILE)
    findBestRotation(img)

test()

# Parse one word of text
import cv
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from heapq import *
import copy

import itl_char

DEBUG = True
IMAGE_FILE = 'images/word1.png'
[IMAGE_NAME, EXTENSION] = IMAGE_FILE.split('.')

def shortestPath(img, startPixel):
    q, visited = [(0,startPixel,[])], set()
    while q:
        (dist, pixel, oldpath) = heappop(q)
        path = copy.copy(oldpath)
        if pixel not in visited:
            visited.add(pixel)
            path.append(pixel)
            if pixel[1] >= img.shape[0]-1:
                return (dist, path)
            light = img[pixel[1]][pixel[0]]
            for i in range(3):
                if i == 0:
                    nxt = (pixel[0], pixel[1]+1)
                elif i == 1 and pixel[0] > 0:
                    nxt = (pixel[0]-1, pixel[1])
                elif i == 2 and pixel[0] < img.shape[1]-1:
                    nxt = (pixel[0]+1, pixel[1])
                #print light
                #print img[nxt[1]][nxt[0]]
                d = abs(int(img[nxt[1]][nxt[0]]) - int(light))
                if nxt not in visited:
                    heappush(q, (dist+d, nxt, path))

# Word image img
def parseWord(img, returnBounds=False):
    img1 = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    img1 = cv2.resize(img1, (img1.shape[1]*6, img1.shape[0]*6))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    img1 = cv2.morphologyEx(img1, cv.CV_MOP_CLOSE, element)
    # img1 = cv2.equalizeHist(img1, img1)
    img2 = copy.copy(img1)
    cv2.imshow('Agg', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            img2[i][j] = max(0,img2[i][j] - np.random.random_sample()*12)
    cv2.imshow('Agg2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    for i in xrange(img2.shape[1]/10):
        startPixel = (i*10,0)
        dist, path = shortestPath(img2, startPixel)
        for point in path:
            cv2.circle(img1, (point[0],point[1]), 0, (0,0,0))
    cv2.imshow('Agg', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    # img1 = cv2.GaussianBlur(img1, (15, 15), 0)
    #unused, img1 = cv2.threshold(img1, 160, 255, cv.CV_THRESH_BINARY_INV)
    # img1 = cv2.Laplacian(img1, cv2.CV_8U)
    # img1 = cv2.Sobel(img1,cv2.CV_8U,1,1,ksize=7)
    # img1 = cv2.dilate(img1, (2, 2), 1)
    #if DEBUG:
        #cv2.imwrite(IMAGE_NAME + '-thresh.' + EXTENSION, img1)

    # Blur in the horizontal direction to get lines
    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    #img1 = cv2.morphologyEx(img1, cv.CV_MOP_CLOSE, element)
    #if DEBUG:
        #cv2.imshow('WordMorph', img1)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # cv2.imwrite(IMAGE_NAME + '-morph.' + EXTENSION, img1)

    # Use RETR_EXTERNAL to remove boxes that are completely contained by the word
    contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundRects = []
    for i in xrange(len(contours)):
        contourPoly = cv2.approxPolyDP(contours[i], 3, True)
        boundRect = cv2.boundingRect(contourPoly)
        #boundRect = (int(round((boundRect[0]*1.0)/10)), int(round((boundRect[1]*1.0)/10)),
        #             int(round((boundRect[2]*1.0)/10)), int(round((boundRect[3]*1.0)/10)))
        boundRects.append(boundRect)

    # Pad the character bounding boxes
    boundRects = sorted(boundRects, key=lambda x: x[0])
    pad = 1
    adjustedRects = []
    for rect in boundRects:
        adjustedRect = (rect[0] - pad, rect[1] - pad, rect[2] + pad * 2, rect[3] + pad * 2)
        adjustedRects.append(adjustedRect)

    #if DEBUG:
     #   for rect in adjustedRects:
      #      cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
       # cv2.imshow('Agg', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #cv2.imwrite(IMAGE_NAME + '-bounds.' + EXTENSION, img)
        #print '%d chars found in %s' % (len(adjustedRects), IMAGE_FILE)

    # print '%d chars found' % len(adjustedRects)
    #for rect in adjustedRects:
        #print rect[0]
        #print rect[1]
        #cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
    # cv2.imshow('Agg', img)
    # cv2.waitKey(0)

    # Extract characters from word
    chars = []
    for rect in adjustedRects:
        [x, y, w, h] = rect
        char = img[y:(y+h), x:(x+w)]
        chars.append(char)

    if returnBounds:
        return chars

    latex = []
    for i in xrange(len(chars)):
        char = chars[i]
        charLatex = itl_char.parseCharacter(char)
        latex.append(charLatex)
        # if DEBUG:
        #     cv2.imshow('%d' % i, char)
        #     cv2.waitKey(0)

    return latex

def test():
    global DEBUG
    DEBUG = True
    #for IMAGE_FILE in ['images/word1.png', 'images/word2.png', 'images/word3.png', 'images/word4.png']:
    for IMAGE_FILE in ['images/word3.png']:
        img = cv2.imread(IMAGE_FILE)
        parseWord(img)

if __name__ == "__main__":
    test()

# Parse one paragraph of text
import cv
import cv2
import numpy as np
import sys

import itl_equation

DEBUG = False
IMAGE_FILE = 'images/eqblock1.png'
[IMAGE_NAME, EXTENSION] = IMAGE_FILE.split('.')

STD_WIDTH = 700

def constructLatex(boundRects, img):
    latex = ""
    topIndex = -1
    while len(boundRects) > 0:
        for i in range(len(boundRects)):
            if topIndex == -1 or boundRects[i][1] < topIndex:
                topIndex = i
        [x, y, w, h] = boundRects[i]
        equation = img[y:(y+h), x:(x+w)]
        cv2.imshow('Equation', equation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        latex += itl_equation.parseEquation(equation)
        latex += "\\"
        del(boundRects[topIndex])
    return latex  

# Paragraph image img
def parseEqBlock(img, returnBounds=False):
    # Standardize paragraph to have a width of STD_WIDTH pixels
    w = img.shape[1]
    scaleFactor = float(STD_WIDTH) / w
    img = cv2.resize(img, (0,0), fx=scaleFactor, fy=scaleFactor)

    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_lap = cv2.Laplacian(blur, cv2.CV_8U)
    unused, img_threshold = cv2.threshold(img_lap, 0, 255, cv.CV_THRESH_OTSU + cv.CV_THRESH_BINARY)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-thresh.' + EXTENSION, img_threshold)

    # Blur in the horizontal direction to get lines
    print returnBounds
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 10) if returnBounds else (40, 10))
    morphed = cv2.morphologyEx(img_threshold, cv.CV_MOP_CLOSE, element)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-morph.' + EXTENSION, morphed)

    # Use RETR_EXTERNAL to remove boxes that are completely contained by the line
    contours, hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundRects = []
    for i in xrange(len(contours)):
        contourPoly = cv2.approxPolyDP(contours[i], 3, True)
        boundRect = cv2.boundingRect(contourPoly)
        boundRects.append(boundRect)

    # Filter bounding rectangles that are not an entire line
    # Take the maximum height among all bounding boxes
    # Remove those boxes that have height less than 25% of the maximum
    maxHeight = -1
    for rect in boundRects:
        maxHeight = max(rect[3], maxHeight)
    heightThresh = .25 * maxHeight
    boundRects = [rect for rect in boundRects if rect[3] > heightThresh]
    boundRects = sorted(boundRects, key=lambda x: x[1])

    # We assume that the image is wrapped fairly tightly around the paragraph
    # We are only concerned with finding line boundaries, so extend the y-coordinate as
    # far up as possible, and pad the width as well
    w = img.shape[0]
    adjustedRects = []
    bottomPad = 2
    rightPad = 4
    topPad = 2
    for rect in boundRects:
        h = rect[3] + topPad + bottomPad
        adjustedRect = (rect[0], rect[1] - topPad, max(rect[2] + rightPad, w), h)
        adjustedRects.append(adjustedRect)

    for rect in adjustedRects:
        cv2.rectangle(morphed, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, thickness=-1)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-morph2.' + EXTENSION, morphed)
    contours, hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundRects = []
    for i in xrange(len(contours)):
        contourPoly = cv2.approxPolyDP(contours[i], 3, True)
        boundRect = cv2.boundingRect(contourPoly)
        boundRects.append(boundRect)

    # Extract lines from paragraph
    equations = []
    for rect in boundRects:
        [x, y, w, h] = rect
        equation = img[y:(y+h), x:(x+w)]
        cv2.imshow('Equation', equation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        equations.append(equation)

    # Returns list of lines
    # Each line is a list of words
    # Each word is a list of chars
    # Each char is a tuple ('latexString', itl_char.CODE)
    return constructLatex(boundRects, img)


def test():
    global DEBUG
    DEBUG = True
    img = cv2.imread(IMAGE_FILE)
    parseEqBlock(img)

if __name__ == "__main__":
    test()

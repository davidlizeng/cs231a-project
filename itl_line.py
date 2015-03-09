# Parse one line of text
import cv
import cv2
import numpy as np
import sys

DEBUG = False
IMAGE_FILE = 'images/line1.png'
[IMAGE_NAME, EXTENSION] = IMAGE_FILE.split('.')

# Line image img
def parseLine(img):
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    # blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_lap = cv2.Laplacian(img_gray, cv2.CV_8U)
    unused, img_threshold = cv2.threshold(img_lap, 0, 255, cv.CV_THRESH_BINARY)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-thresh.' + EXTENSION, img_threshold)

    # Blur in the horizontal direction to get lines
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
    morphed = cv2.morphologyEx(img_threshold, cv.CV_MOP_CLOSE, element)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-morph.' + EXTENSION, morphed)

    # Use RETR_EXTERNAL to remove boxes that are completely contained by the word
    contours, hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundRects = []
    for i in xrange(len(contours)):
        contourPoly = cv2.approxPolyDP(contours[i], 3, True)
        boundRect = cv2.boundingRect(contourPoly)
        boundRects.append(boundRect)

    # We assume that the image is wrapped fairly tightly around the line
    # We are only concerned with finding word boundaries, so extend the x-coordinate as
    # far left as possible, and then extend the height to maximum
    h = img.shape[0]
    adjustedRects = []
    currentLeftX = 0
    rightPad = 2
    for rect in boundRects:
        x = currentLeftX
        w = rect[2] + (rect[0] - currentLeftX) + rightPad
        adjustedRect = (x, 0, w, h)
        adjustedRects.append(adjustedRect)
        currentLeftX = rect[0] + rect[2] + rightPad

    if DEBUG:
        for rect in adjustedRects:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
        cv2.imwrite(IMAGE_NAME + '-bounds.' + EXTENSION, img)
        print '%d words found in %s' % (len(adjustedRects), IMAGE_FILE)

    # Extract words from paragraph
    words = []
    for word in adjustedRects:
        [x, y, w, h] = rect
        word = img[y:(y+h), x:(x+w)]
        words.append(word)

    # for word in words:
    #     # TODO do something here
    #     if DEBUG:
    #         cv2.imshow('Word', word)
    #         cv2.waitKey(0)

    return boundRects


def test():
    global DEBUG
    DEBUG = True
    img = cv2.imread(IMAGE_FILE)
    parseLine(img)

if __name__ == "__main__":
    test()
# Parse one word of text
import cv
import cv2
import numpy as np
import sys

import itl_char

IMAGE_FILES = ['images/word1.png', 'images/word2.png', 'images/word3.png', 'images/word4.png']

DEBUG = False
IMAGE_FILE = ''
[IMAGE_NAME, EXTENSION] = ['', '']

# Word image img
def parseWord(img, returnBounds=False):
    img1 = cv2.cvtColor(img, cv.CV_BGR2GRAY)

    if returnBounds:
        # TRAINING parameters, with nicely spaced characters
        unused, img1 = cv2.threshold(img1, 200, 255, cv.CV_THRESH_BINARY_INV)
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
        img1 = cv2.dilate(img1, (2, 2), 1)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img1 = cv2.morphologyEx(img1, cv.CV_MOP_CLOSE, element)
        # cv2.imshow('MORPH', img1)
        # cv2.waitKey(0)
    else:
        img1 = cv2.GaussianBlur(img1, (1, 1), 0)
        # img1 = cv2.Laplacian(img1, cv2.CV_8U)
        # img1 = cv2.equalizeHist(img1)
        unused, img1 = cv2.threshold(img1, 200, 255, cv.CV_THRESH_BINARY_INV)
        # img1 = cv2.dilate(img1, (1, 1), 1)
        if DEBUG:
            cv2.imwrite(IMAGE_NAME + '-thresh.' + EXTENSION, img1)
        # Blur in the horizontal direction to get lines
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        img1 = cv2.morphologyEx(img1, cv.CV_MOP_CLOSE, element)

    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-morph.' + EXTENSION, img1)

    # Use RETR_EXTERNAL to remove boxes that are completely contained by the word
    contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundRects = []
    for i in xrange(len(contours)):
        contourPoly = cv2.approxPolyDP(contours[i], 3, True)
        boundRect = cv2.boundingRect(contourPoly)
        boundRects.append(boundRect)

    # Pad the character bounding boxes
    height = img.shape[0]
    width = img.shape[1]
    boundRects = sorted(boundRects, key=lambda x: x[0])
    pad = 1
    adjustedRects = []
    minX = width
    maxX = 0
    for rect in boundRects:
        adjustedRect = (rect[0] - pad, rect[1] - pad, rect[2] + pad * 2, rect[3] + pad * 2)
        adjustedRects.append(adjustedRect)
        minX = min(minX, adjustedRect[0])
        maxX = max(maxX, adjustedRect[0] + adjustedRect[2])

    if DEBUG:
        # for rect in adjustedRects:
        #     cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
        cv2.imwrite(IMAGE_NAME + '-bounds.' + EXTENSION, img)
        print '%d char bounding boxes initially found in %s' % (len(adjustedRects), IMAGE_FILE)

    # for rect in adjustedRects:
    #     cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
    # cv2.imshow('IMG', img)
    # cv2.waitKey(0)

    # Extract characters from word
    chars = []
    for rect in adjustedRects:
        [x, y, w, h] = rect
        char = img[0:height, x:(x+w)]
        chars.append(char)
    # currentX = minX
    # MIN_CHAR_WIDTH = 3
    # MAX_CHAR_WIDTH = 20
    # CHAR_BOX_STEP = 1
    # while width - currentX >= MIN_CHAR_WIDTH:
    #     bestScore = 0 # Start this at threshold!
    #     bestVal = None
    #     bestWidth = 0
    #     for width in xrange(MIN_CHAR_WIDTH, MAX_CHAR_WIDTH + 1, CHAR_BOX_STEP):
    #         rightX = currentX + width
    #         charSlice = img[0:height, currentX:rightX]
    #         val, score = itl_char.parseCharacter(charSlice, getScore=True)
    #         if score > bestScore:
    #             bestScore = score
    #             bestWidth = width
    #             bestVal = val
    #     print bestWidth, bestScore, bestVal
    #     # Add to chars array
    #     currentX += bestWidth

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
    global DEBUG, IMAGE_FILE, IMAGE_NAME, EXTENSION
    DEBUG = True
    for IMAGE_FILE in IMAGE_FILES:
        [IMAGE_NAME, EXTENSION] = IMAGE_FILE.split('.')
        img = cv2.imread(IMAGE_FILE)
        parseWord(img)

if __name__ == "__main__":
    test()

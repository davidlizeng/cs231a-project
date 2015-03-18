# Parse one word of text
import cv
import cv2
import numpy as np
import sys

import itl_char

IMAGE_FILES = ['images/word1.png', 'images/word2.png', 'images/word3.png', 'images/word5.png', 'images/word4.png']

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
    # chars = []
    # for rect in adjustedRects:
    #     [x, y, w, h] = rect
    #     char = img[0:height, x:(x+w)]
    #     chars.append(char)
    latex = []
    img = itl_char.getCroppedImage(itl_char.getBinaryImage(img))
    width = img.shape[1]
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    columnSums = np.sum(img, axis=0)
    currentX = 0
    MIN_WIDTH = 2
    MIN_CHAR_WIDTH = 4
    MAX_CHAR_WIDTH = 15
    THRESHOLD = .8
    while width - currentX >= MIN_WIDTH:
        if columnSums[currentX] == 0 or columnSums[currentX + 1] == 0:
            currentX += 1
            continue
        start = MIN_CHAR_WIDTH
        end = MAX_CHAR_WIDTH
        if (currentX + 2 < width and columnSums[currentX + 2] == 0) or\
           (currentX + 3 < width and columnSums[currentX + 3] == 0):
            start = MIN_WIDTH
            end = MIN_CHAR_WIDTH + 1
        scores = [False] * (MAX_CHAR_WIDTH + 1)
        values = [None] * (MAX_CHAR_WIDTH + 1)
        brokeSpace = False
        goodCharScore = False
        for w in xrange(start, end):
            rightX = currentX + w
            charSlice = img[0:height, currentX:rightX]
            # print charSlice
            val, score = itl_char.parseCharacter(charSlice, getScore=True, isBinary=True)
            scores[w] = score
            if score > THRESHOLD and w >= MIN_CHAR_WIDTH:
                goodCharScore = True
            values[w] = val
            if rightX >= width or columnSums[rightX] == 0:
                brokeSpace = True
                break

        maxRunLength = 0
        maxRunBestScore = 0.0
        maxRunBestWidth = 0
        currentRunLength = 0
        currentRunBestScore = 0.0
        currentRunBestWidth = 0
        if brokeSpace:
            maxRunBestWidth = w
            maxRunBestScore = scores[w]
            bestVal = values[w]
        else:
            for w in xrange(start, end + 1):
                if scores[w] < THRESHOLD or values[w][0] in '.,:;':
                    currentRunLength = 0
                    currentRunBestScore = 0.0
                    continue
                currentRunLength += 1
                if scores[w] > currentRunBestScore:
                    currentRunBestScore = scores[w]
                    currentRunBestWidth = w
                if currentRunLength > maxRunLength or\
                   (currentRunLength == maxRunLength and currentRunBestScore > maxRunBestScore):
                    maxRunLength = currentRunLength
                    maxRunBestScore = currentRunBestScore
                    maxRunBestWidth = currentRunBestWidth

        if maxRunBestWidth == 0:
            break

        bestVal = values[maxRunBestWidth]
        # print '*****', maxRunBestWidth, maxRunBestScore, bestVal
        # Add to chars array
        currentX += maxRunBestWidth
        latex.append(bestVal)

    if returnBounds:
        return chars

    # latex = []
    # for i in xrange(len(chars)):
    #     char = chars[i]
    #     charLatex = itl_char.parseCharacter(char)
    #     latex.append(charLatex)
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

import cv
import cv2
import numpy as np
import sys
import itl_paragraph

IMAGE_FILE = 'images/paper2.png'
[IMAGE_NAME, EXTENSION] = IMAGE_FILE.split('.')

def parsePaper(img):
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    img_sob = cv2.Sobel(img_gray,cv2.CV_8U,1,1,ksize=7)
    img_lap = cv2.Laplacian(img_gray, cv2.CV_8U)
    unused, img_threshold = cv2.threshold(img_sob, 0, 255, cv.CV_THRESH_OTSU + cv.CV_THRESH_BINARY)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-thresh.' + EXTENSION, img_threshold)

    # Blur to get paragraph blocks
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    morphed = cv2.morphologyEx(img_threshold, cv.CV_MOP_CLOSE, element)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-morph.' + EXTENSION, morphed)

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
    headerRects = []
    maxLineHeight = 0
    maxLineHeight2 = 0
    for rect in boundRects:
        if rect[2]*rect[3] > 250:
            if (rect[0]*1.0)/paperWidth > 0.225:
                equationRects.append(rect)
            else:
                lineRects = itl_paragraph.parseParagraph(img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]])
                lineSize = (rect[3]*1.0)/len(lineRects)
                if lineSize > maxLineHeight:
                    maxLineHeight2 = maxLineHeight
                    maxLineHeight = lineSize
                elif lineSize > maxLineHeight2:
                    maxLineHeight2 = lineSize
    for rect in boundRects:
        if rect[2]*rect[3] > 250:
            if not (rect[0]*1.0)/paperWidth > 0.225:
                isHeader = False
                lineRects = itl_paragraph.parseParagraph(img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]])
                lineSize = (rect[3]*1.0)/len(lineRects)
                print lineSize
                if lineSize >= maxLineHeight and lineSize >= maxLineHeight2 * 1.05:
                    headerRects.append(rect)
                else:
                    textRects.append(rect)

    if DEBUG:
        for rect in textRects:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
        for rect in equationRects:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0))
        for rect in headerRects:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255))
        cv2.imwrite(IMAGE_NAME + '-bounds.' + EXTENSION, img)
        print '%d equations found in %s' % (len(equationRects), IMAGE_FILE)
        print '%d texts found in %s' % (len(textRects), IMAGE_FILE)
        print '%d headers found in %s' % (len(headerRects), IMAGE_FILE)

    # Extract lines from paragraph
    paragraphs = []
    for rect in boundRects:
        [x, y, w, h] = rect
        paragraph = img[y:(y+h), x:(x+w)]
        paragraphs.append(paragraph)

    for paragraph in paragraphs:
        # TODO do something here
        if DEBUG:
            cv2.imshow('Paragraph', paragraph)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return boundRects   

def test():
    global DEBUG
    DEBUG = True
    img = cv2.imread(IMAGE_FILE)
    parsePaper(img)

test()

import cv
import cv2
import numpy as np
import sys
import itl_paragraph
import rotation_fix
import itl_eqblock

IMAGE_FILE = 'images/paper1.png'
OUTPUT_FILE = 'genLatex.tex'
[IMAGE_NAME, EXTENSION] = IMAGE_FILE.split('.')
TEXT = 1
EQUATION = 2
DEBUG = False

def displayImage(img):
    cv2.imshow('Picture', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def initializeLatexPaper():
    latexPaper = ""
    latexPaper += "\\documentclass[11pt]{article} \n"
    latexPaper += "\usepackage{algorithm, algpseudocode, amsmath, amsfonts, cite, graphicx, icomma, multirow, url, xspace, tikz, qtree} \n"
    latexPaper += "\\begin{document} \n"
    return latexPaper

def addLatex(latexPaper, latex, paragraphType):
    if paragraphType == EQUATION:
        latexPaper += "\\begin{align*} \n"
    latexPaper += latex
    if paragraphType == TEXT:
        latexPaper += "\\\\ \\\\ \n"
    elif paragraphType == EQUATION:
        latexPaper += "\\end{align*} \n"
    return latexPaper

def constructLatex(paragraphs, paragraphBoxes, paragraphTypes):
    latexPaper = initializeLatexPaper()
    while len(paragraphs) > 0:
        topIndex = -1
        for i in range(len(paragraphs)):
            if topIndex == -1 or paragraphBoxes[i][1] < paragraphBoxes[topIndex][1]:
                topIndex = i
        if paragraphTypes[topIndex] == 1:
            latex = itl_paragraph.parseParagraph(paragraphs[topIndex])
        elif paragraphTypes[topIndex] == 2:
            latex = itl_eqblock.parseEqBlock(paragraphs[topIndex])
        latexPaper = addLatex(latexPaper, latex, paragraphTypes[topIndex])
        del(paragraphs[topIndex])
        del(paragraphTypes[topIndex])
        del(paragraphBoxes[topIndex])

    latexPaper += "\\end{document}"
    return latexPaper

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
    for rect in boundRects:
        if rect[2]*rect[3] > 250:
            if (rect[0]*1.0)/paperWidth > 0.225:
                equationRects.append(rect)
            else:
                textRects.append(rect)

    if DEBUG:
        for rect in textRects:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
        for rect in equationRects:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0))
        cv2.imwrite(IMAGE_NAME + '-bounds.' + EXTENSION, img)
        print '%d equations found in %s' % (len(equationRects), IMAGE_FILE)
        print '%d texts found in %s' % (len(textRects), IMAGE_FILE)

    # Get paragraphs
    paragraphs = []
    paragraphBoxes = []
    paragraphTypes = []
    for rect in textRects:
        [x, y, w, h] = rect
        paragraph = img[y:(y+h), x:(x+w)]
        paragraphs.append(paragraph)
        paragraphBoxes.append([x, y, w, h])
        paragraphTypes.append(TEXT)
    for rect in equationRects:
        [x, y, w, h] = rect
        paragraph = img[y:(y+h), x:(x+w)]
        paragraphs.append(paragraph)
        paragraphBoxes.append([x, y, w, h])
        paragraphTypes.append(EQUATION)

    return constructLatex(paragraphs, paragraphBoxes, paragraphTypes)

def createLatexFile(imgFile, outFile):
    img = cv2.imread(imgFile)
    img = rotation_fix.findBestRotation(img)
    latexPaper = parsePaper(img)
    f = open(outFile, 'w+')
    f.write(latexPaper)

def test():
    global DEBUG
    DEBUG = True
    img = cv2.imread(IMAGE_FILE)
    print parsePaper(img)

createLatexFile(IMAGE_FILE, OUTPUT_FILE)

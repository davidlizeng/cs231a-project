import cv
import cv2
import numpy as np
import sys
import math

DEBUG = False
IMAGE_FILE = 'images/equation2.png'
[IMAGE_NAME, EXTENSION] = IMAGE_FILE.split('.')

symbol_dict = {
    'e'         : ('c', ('subscr', 'supscr')),
    'f'         : ('a', ('subscr', 'supscr')),
    'i'         : ('c', ('subscr', 'supscr')),
    'n'         : ('c', ('subscr', 'supscr')),
    'x'         : ('c', ('subscr', 'supscr')),
    '2'         : ('a', ('subscr', 'supscr')),
    '\\pi'      : ('c', ('subscr', 'supscr')),
    '\\infty'   : ('c', ('subscr', 'supscr')),
    '='         : ('c', ()),
    '\\sum'     : ('c', ('above', 'below')),
    '|'         : ('c', ('subscr', 'supscr')),
    '\\langle'  : ('c', ('subscr', 'supscr')),
    '\\rangle'  : ('c', ('subscr', 'supscr')),
    '-'         : ('c', ('above', 'below')),
    '\\sqrt'    : ('c', ('subexp',)),
    ','         : ('d', ()),
}

class Symbol:
    def __init__(self, x, y, w, h, t, r):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.type = t
        self.range = r

    def area(self):
        return self.w * self.h

    def minX(self):
        return self.x

    def minY(self):
        return self.y

    def maxX(self):
        return self.x + self.w

    def maxY(self):
        return self.y + self.h

    def width(self):
        return self.w

    def height(self):
        return self.h

    def isAbove(self, other):
        c = other.centroid()
        return self.minX() <= c[0] <= self.maxX() and \
            self.supThreshold() <= c[1]

    def isBelow(self, other):
        c = other.centroid()
        return self.minX() <= c[0] <= self.maxX() and \
            self.subThreshold() >= c[1]

    def isSubexp(self, other):
        return self.minX() <= other.minX() and self.minY() <= other.minY() and \
            self.maxX() >= other.maxX() and self.maxY() >= other.maxY()

    def isSupscr(self, other):
        return self.centroid()[0] <= other.minX() and \
            self.supThreshold() <= other.centroid()[1]

    def isSubscr(self, other):
        return self.centroid()[0] <= other.minX() and \
            self.subThreshold() >= other.centroid()[1]

    def dominates(self, other):
        inDomRegion = False
        for region in self.range:
            inDomRegion |= Symbol.domFunc[region](self, other)
        return inDomRegion

    def distance(self, other):
        c1 = self.centroid()
        c2 = other.centroid()
        eucDist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        if self.dominates(other) or other.dominates(self):
            return eucDist*0.5
        else:
            return eucDist


    domFunc = {'above': isAbove, 'below': isBelow, 'subexp': isSubexp, \
        'supscr': isSupscr, 'subscr': isSubscr}


class AscSymbol(Symbol):
    def __init__(self, x, y, w, h, t, r):
        Symbol.__init__(self, x, y, w, h, t, r)

    def supThreshold(self):
        return self.y + 0.8*self.h

    def subThreshold(self):
        return self.y + 0.2*self.h

    def centroid(self):
        return (self.x + 0.5*self.w, self.y + 0.33*self.h)

class DesSymbol(Symbol):
    def __init__(self, x, y, w, h, t, r):
        Symbol.__init__(self, x, y, w, h, t, r)

    def supThreshold(self):
        return self.y + 0.9*self.h

    def subThreshold(self):
        return self.y + 0.6*self.h

    def centroid(self):
        return (self.x + 0.5*self.w, self.y + 0.66*self.h)

class CenSymbol(Symbol):
    def __init__(self, x, y, w, h, t, r):
        Symbol.__init__(self, x, y, w, h, t, r)

    def supThreshold(self):
        return self.y + 0.8*self.h

    def subThreshold(self):
        return self.y + 0.2*self.h

    def centroid(self):
        return (self.x + 0.5*self.w, self.y + 0.5*self.h)

def buildSymbol(text, x, y, w, h):
    attrs = symbol_dict[text]
    if attrs[0] == 'a':
        return AscSymbol(x, y, w, h, attrs[0], attrs[1])
    elif attrs[0] == 'c':
        return CenSymbol(x, y, w, h, attrs[0], attrs[1])
    elif attrs[0] == 'd':
        return DesSymbol(x, y, w, h, attrs[0], attrs[1])
    else:
        print 'unsupported symbol'
        return None


# L should be sorted by x value
def findBaseLine(L, y_center, thresh):
    baseline = []
    for i in xrange(len(L)):
        c = L[i].centroid()
        if abs(c[1] - y_center) < thresh:
            baseline.append(i)
    print baseline
    return baseline


def findMST(L, baseline):
    dists = []
    tree = np.zeros((len(L), len(L)))
    used = set(baseline)
    for i in xrange(len(L)):
        for j in xrange(i+1, len(L)):
            dists.append((L[i].distance(L[j]), i, j))
    sortedDists = sorted(dists)
    for i in xrange(len(baseline) - 1):
        tree[baseline[i]][baseline[i+1]] = 1
        tree[baseline[i+1]][baseline[i]] = 1
    while len(used) < len(L):
        for edge in sortedDists:
            if (edge[1] in used) != (edge[2] in used):
                used.add(edge[1])
                used.add(edge[2])
                tree[edge[1]][edge[2]] = 1
                tree[edge[2]][edge[1]] = 1
                break
    return tree

def findSymbolTree(L, img):
    baseline = findBaseLine(L, img.shape[0]/2.0, img.shape[0]/20.0)
    tree = findMST(L, baseline)
    return tree


# equation image
def parseEquation(img):
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)

    img_lap = cv2.Laplacian(img_gray, cv2.CV_8U)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-lap.' + EXTENSION, img_lap)
    unused, img_threshold = cv2.threshold(img_lap, 0, 255, cv.CV_THRESH_BINARY)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-thresh.' + EXTENSION, img_threshold)
    # Blur in the horizontal direction to get lines
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
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

    # # Filter bounding rectangles that are not an entire line
    # # Take the maximum height among all bounding boxes
    # # Remove those boxes that have height less than 25% of the maximum
    # maxHeight = -1
    # for rect in boundRects:
    #     maxHeight = max(rect[3], maxHeight)
    # heightThresh = .25 * maxHeight
    # boundRects = [rect for rect in boundRects if rect[3] > heightThresh]

    if DEBUG:
        for rect in boundRects:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
        cv2.imwrite(IMAGE_NAME + '-bounds.' + EXTENSION, img)
        print '%d words found in %s' % (len(boundRects), IMAGE_FILE)

    sortedRects = sorted(boundRects, key=lambda x:x[0])
    # words = []
    # for rect in sortedRects:
    #     [x, y, w, h] = rect
    #     word = img[y:(y+h), x:(x+w)]
    #     words.append(word)

    # for word in words:
    #     # TODO do something here
    #     if DEBUG:
    #         cv2.imshow('Word', word)
    #         cv2.waitKey(0)
    johns = ['n', '=', '\\sum', '\\infty', '-', '\\infty', '|', '\\langle',
        'f', ',', '-', '\\sqrt', 'e', 'i', 'n', '2', 'x', '\\pi', '\\rangle',
        '|', '2', '-', '-', '|', '|', 'f', '|', '|', '2'
    ]
    L = []
    for j in xrange(len(johns)):
        L.append(buildSymbol(johns[j], *(sortedRects[j])))

    tree = findSymbolTree(L, img_gray)

    for sym in L:
        c = sym.centroid()
        rounded = (int(round(c[0])), int(round(c[1])))
        cv2.circle(img, rounded, 2, (0, 0, 255))
    for i in xrange(len(L)):
        for j in xrange(i + 1, len(L)):
            if tree[i][j] > 0:
                c1 = L[i].centroid()
                c2 = L[j].centroid()
                r1 = (int(round(c1[0])), int(round(c1[1])))
                r2 = (int(round(c2[0])), int(round(c2[1])))
                cv2.line(img, r1, r2, (0, 0, 255))
    cv2.imwrite(IMAGE_NAME + '-mst.' + EXTENSION, img)

    return boundRects


def test():
    global DEBUG
    DEBUG = True
    img = cv2.imread(IMAGE_FILE)
    parseEquation(img)

if __name__ == "__main__":
    test()

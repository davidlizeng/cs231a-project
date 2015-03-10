import cv
import cv2
import numpy as np
import sys
import math
from collections import deque

DEBUG = False
IMAGE_FILE = 'images/equation4.png'
[IMAGE_NAME, EXTENSION] = IMAGE_FILE.split('.')

symbol_dict = {
    'e'         : ('c', ('subscr', 'supscr')),
    'f'         : ('c', ('subscr', 'supscr')),
    'i'         : ('a', ('subscr', 'supscr')),
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
    '\\frac'    : ('c', ('above', 'below')),
}

class Symbol:
    def __init__(self, x, y, w, h, t, r, k):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.type = t
        self.range = r
        self.key = k

    def perimeter(self):
        return 2*self.w + 2*self.h

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

    def domAbove(self, other):
        c = other.centroid()
        return self.minX() <= c[0] <= self.maxX() and \
            self.supThreshold() >= c[1]

    def domBelow(self, other):
        c = other.centroid()
        return self.minX() <= c[0] <= self.maxX() and \
            self.subThreshold() <= c[1]

    def domSubexp(self, other):
        return self.minX() <= other.minX() and self.minY() <= other.minY() and \
            self.maxX() >= other.maxX() and self.maxY() >= other.maxY()

    def domSupscr(self, other):
        return self.centroid()[0] <= other.minX() and \
            self.supThreshold() >= other.centroid()[1]

    def domSubscr(self, other):
        return self.centroid()[0] <= other.minX() and \
            self.subThreshold() <= other.centroid()[1]

    def dominates(self, other):
        inDomRegion = False
        for region in self.range:
            inDomRegion |= Symbol.domFunc[region](self, other)
        return inDomRegion

    def distance(self, other):
        c1 = self.centroid()
        c2 = other.centroid()
        eucDist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        if other.dominates(self) or self.dominates(other):
            return eucDist*0.9
        else:
            return eucDist

    domFunc = {'above': domAbove, 'below': domBelow, 'subexp': domSubexp, \
        'supscr': domSupscr, 'subscr': domSubscr}


class AscSymbol(Symbol):
    def __init__(self, x, y, w, h, t, r, k):
        Symbol.__init__(self, x, y, w, h, t, r, k)

    def supThreshold(self):
        return self.y + 0.2*self.h

    def subThreshold(self):
        return self.y + 0.8*self.h

    def centroid(self):
        return (self.x + 0.5*self.w, self.y + 0.66*self.h)

class DesSymbol(Symbol):
    def __init__(self, x, y, w, h, t, r, k):
        Symbol.__init__(self, x, y, w, h, t, r, k)

    def supThreshold(self):
        return self.y + 0.1*self.h

    def subThreshold(self):
        return self.y + 0.4*self.h

    def centroid(self):
        return (self.x + 0.5*self.w, self.y + 0.33*self.h)

class CenSymbol(Symbol):
    def __init__(self, x, y, w, h, t, r, k):
        Symbol.__init__(self, x, y, w, h, t, r, k)

    def supThreshold(self):
        return self.y + 0.2*self.h

    def subThreshold(self):
        return self.y + 0.8*self.h

    def centroid(self):
        return (self.x + 0.5*self.w, self.y + 0.5*self.h)

def buildSymbol(key, x, y, w, h):
    attrs = symbol_dict[key]
    if attrs[0] == 'a':
        return AscSymbol(x, y, w, h, attrs[0], attrs[1], key)
    elif attrs[0] == 'c':
        return CenSymbol(x, y, w, h, attrs[0], attrs[1], key)
    elif attrs[0] == 'd':
        return DesSymbol(x, y, w, h, attrs[0], attrs[1], key)
    else:
        print 'unsupported symbol'
        return None

def findDomSymbol(L):
    bestInd = 0
    bestPerim = 0
    for i in xrange(len(L)):
        if L[i].perimeter() > bestPerim:
            bestPerim = L[i].perimeter()
            bestInd = i
    return L[bestInd]

# L should be sorted by x value
def findBaseLine(L):
    domSymbol = findDomSymbol(L)
    y_center = domSymbol.centroid()[1]
    minY = min([s.minY() for s in L])
    maxY = max([s.maxY() for s in L])
    thresh = (maxY - minY)/10.0
    baseline = []
    for i in xrange(len(L)):
        c = L[i].centroid()
        if abs(c[1] - y_center) < thresh:
            baseline.append(i)
    dominated = set()
    for i in xrange(len(baseline)):
        for j in xrange(i+1, len(baseline)):
            if L[baseline[i]].dominates(L[baseline[j]]):
                dominated.add(j)
            if L[baseline[j]].dominates(L[baseline[i]]):
                dominated.add(i)
    dominated = sorted(dominated, reverse=True)
    for d in dominated:
        baseline.pop(d)
    print 'baseline', baseline
    return baseline


def findMST(L, baseline):
    dists = []
    tree = [[] for i in xrange(len(L))]
    used = set(baseline)
    for i in xrange(len(L)):
        for j in xrange(i+1, len(L)):
            dists.append((L[i].distance(L[j]), i, j))
    sortedDists = sorted(dists)
    for i in xrange(len(baseline) - 1):
        tree[baseline[i]].append(baseline[i+1])
        tree[baseline[i+1]].append(baseline[i])
    while len(used) < len(L):
        for edge in sortedDists:
            if (edge[1] in used) != (edge[2] in used):
                used.add(edge[1])
                used.add(edge[2])
                tree[edge[1]].append(edge[2])
                tree[edge[2]].append(edge[1])
                break
    return tree

def findSymbolTree(L):
    # y_center = 0
    # thresh = 0
    # if len(L) > 1:
    #     minY = min([s.minY() for s in L])
    #     maxY = max([s.maxY() for s in L])
    #     expHeight = maxY - minY
    #     y_center = minY + expHeight/2.0
    #     thresh = expHeight/10.0
    # else:
    #     y_center = L[0].centroid()[1]
    #     thresh = 0.1

    baseline = findBaseLine(L)
    tree = findMST(L, baseline)
    print 'tree', tree
    return tree, baseline


class LaTeXNode:
    def __init__(self, cmd='', sup=None, sub=None, args=[]):
        self.cmd = cmd
        self.args = args
        self.sup = sup
        self.sub = sub

    def str(self):
        return ''.join(self.strList())

    def strList(self):
        strings = [self.cmd]
        for arg in self.args:
            strings.append('{')
            strings.extend(arg.strList())
            strings.append('}')
        if self.sup == None and self.sub == None:
            strings.append(' ')
        if self.sup != None:
            strings.append('^{')
            strings.extend(self.sup.strList())
            strings.append('}')
        if self.sub != None:
            strings.append('_{')
            strings.extend(self.sub.strList())
            strings.append('}')
        return strings

class ParentNode:
    def __init__(self, children=[]):
        self.children = children

    def str(self):
        return ''.join(self.strList())

    def strList(self):
        strings = []
        for child in self.children:
            strings.extend(child.strList())
        return strings

def findConnectedComponent(L, source, tree, bset):
    agg = [source]
    cc = [L[source]]
    q = deque()
    q.append(source)
    used = bset.copy()
    used.add(source)
    ind = 0
    while len(q) > 0:
        ind = q.popleft()
        for nbr in tree[ind]:
            if nbr not in used:
                used.add(nbr)
                agg.append(nbr)
                cc.append(L[nbr])
                q.append(nbr)
    print 'agg', agg
    return sorted(cc, key=lambda x: (x.minX(), x.minY()))


def buildLaTeXNode(L, root, children, nodes):
    sup = None
    sub = None
    args = []
    cmd = L[root].key
    if L[root].key == '-':
        if len(children) == 0:
            pass
        elif len(children) == 2:
            cmd = '\\frac'
            if L[root].domAbove(L[children[0]]):
                args = nodes
            elif L[root].domBelow(L[children[0]]):
                args.append(nodes[1])
                args.append(nodes[0])
            else:
                print 'unexpected child location', cmd
        else:
            print 'unexpected number of children', cmd
    elif L[root].key == '\\sum':
        if len(children) <= 2:
            for i in xrange(len(children)):
                if L[root].domAbove(L[children[i]]):
                    sup = nodes[i]
                elif L[root].domBelow(L[children[i]]):
                    sub = nodes[i]
        else:
            print 'unexpected number of children', cmd
    elif L[root].key == '\\sqrt':
        if len(children) == 0:
            pass
        elif len(children) == 1:
            if L[root].domSubexp(L[children[0]]):
                args.append(nodes[0])
            else:
                print 'unexpected child location', cmd
        else:
            print 'unexpected number of children', cmd
    elif L[root].key == '=' or L[root].key == ',':
        if len(children) != 0:
            print 'unexpected number of children', cmd
    else:
        if len(children) <= 2:
            for i in xrange(len(children)):
                if L[root].domSupscr(L[children[i]]):
                    sup = nodes[i]
                elif L[root].domSubscr(L[children[i]]):
                    sub = nodes[i]
    return LaTeXNode(cmd, sup, sub, args)



def buildLaTeXTree(L):
    baseNodes = []
    tree, baseline = findSymbolTree(L)
    bset = set(baseline)
    for i in baseline:
        nbrs = tree[i]
        childNodes = []
        childInds = []
        for nbr in nbrs:
            if nbr not in bset:
                cc = findConnectedComponent(L, nbr, tree, bset)
                childNodes.append(buildLaTeXTree(cc))
                childInds.append(nbr)
        baseNodes.append(buildLaTeXNode(L, i, childInds, childNodes))
    return ParentNode(baseNodes)


# handle special symbols cant can't be detected
# = sign v. - sign v. \frac line
def handleSpecialCases(L):
    newL = []
    i = 0
    while True:
        if i + 1 < len(L):
            if L[i].key == '-' and L[i + 1].key == '-' and \
                    L[i].domBelow(L[i+1]) and L[i+1].domAbove(L[i]) and \
                    1.0*abs(L[i].width() - L[i+1].width())/(L[i].width() + L[i+1].width()) < 0.1:
                equal = buildSymbol('=', L[i].minX(), L[i].minY(), L[i].width(), \
                    L[i+1].maxY() - L[i].minY())
                newL.append(equal)
                i += 2
            else:
                newL.append(L[i])
                i += 1
        elif i == len(L) - 1:
            newL.append(L[i])
            i += 1
        else:
            break
    return newL

# equation image
def parseEquation(img):
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    print img_gray.shape
    img_lap = cv2.Laplacian(img_gray, cv2.CV_8U)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-lap.' + EXTENSION, img_lap)
    unused, img_threshold = cv2.threshold(img_lap, 0, 255, cv.CV_THRESH_BINARY)
    if DEBUG:
        cv2.imwrite(IMAGE_NAME + '-thresh.' + EXTENSION, img_threshold)
    # Blur in the horizontal direction to get lines
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
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

    sortedRects = sorted(boundRects, key=lambda x:(x[0], x[1]))
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
    johns = ['n', '-', '-', '\\sum', '\\infty', '-', '\\infty', '|', '\\langle',
        'f', ',', '-', '\\sqrt', 'e', 'i', '2', 'n', '\\pi', 'x', '\\rangle',
        '|', '2', '-', '-', '|', '|', 'f', '|', '|', '2'
    ]
    L = []
    for j in xrange(len(johns)):
        L.append(buildSymbol(johns[j], *(sortedRects[j])))
    L = handleSpecialCases(L)
    death = [l.key for l in L]
    print death
    tree, baseline = findSymbolTree(L)

    for i in xrange(len(L)):
        c = L[i].centroid()
        rounded = (int(round(c[0])), int(round(c[1])))
        if i in baseline:
            cv2.circle(img, rounded, 3, (255, 0, 0), 2)
        else:
            cv2.circle(img, rounded, 3, (0, 0, 255), 2)
    for i in xrange(len(L)):
        for j in tree[i]:
            c1 = L[i].centroid()
            c2 = L[j].centroid()
            r1 = (int(round(c1[0])), int(round(c1[1])))
            r2 = (int(round(c2[0])), int(round(c2[1])))
            if i in baseline and j in baseline:
                cv2.line(img, r1, r2, (255, 0, 0))
            else:
                cv2.line(img, r1, r2, (0, 0, 255))
    cv2.circle(img, (0,0) , 3, (255, 0, 0), 2)
    cv2.imwrite(IMAGE_NAME + '-mst.' + EXTENSION, img)

    node = buildLaTeXTree(L)
    print node.str()

    return boundRects


def test():
    global DEBUG
    DEBUG = True
    img = cv2.imread(IMAGE_FILE)
    parseEquation(img)

if __name__ == "__main__":
    test()

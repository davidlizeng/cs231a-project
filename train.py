# Train character data

import cv
import cv2
import numpy as np

import itl_char
import itl_paragraph

PATH = 'training/'

DICTIONARY_FILE = PATH + 'dictionary.txt'

# Name training images as 'train#.png', where # is the image number.
# Corresponding mapping file is 'map#.txt'. Map file should just be one line.
# Follow examples to add more training images.
NUM_IMAGES = 5

CHAR_BOX = (20, 20)

def readMapFile(filename):
    infile = open(filename, 'r')
    data = []
    for line in infile:
        d = line.split()
        data += map(int, d)
    return data

def trainImage(index):
    imageFile = PATH + ('train%d.png' % index)
    mapFile = PATH + ('map%d.txt' % index)
    nums = readMapFile(mapFile)
    img = cv2.imread(imageFile)
    charBounds = itl_paragraph.parseParagraph(img, returnBounds=True)
    if len(charBounds) != len(nums):
        print 'ERROR: Image #%d, len(mapFile) = %d, len(detectedChars) = %d' %\
            (index, len(nums), len(charBounds))

    samples = np.empty((0, 400))
    for charBound in charBounds:
        charBound = cv2.resize(charBound, CHAR_BOX)
        charBound = cv2.cvtColor(charBound,cv2.COLOR_BGR2GRAY)
        charBound = np.float32(charBound.reshape((1, 400)))
        charBound = charBound / np.linalg.norm(charBound)
        samples = np.append(samples, charBound, 0)
    nums = np.array(nums, np.int)
    nums = nums.reshape((nums.size, 1))
    return (nums, samples)


def trainImages(dictionary):
    nums = np.empty((0, 1))
    samples = np.empty((0, 400))
    for i in xrange(1, NUM_IMAGES + 1):
        (n, s) = trainImage(i)
        nums = np.append(nums, n, 0)
        samples = np.append(samples, s, 0)
        print 'Loaded train%d.png' % i
    np.savetxt('training/samples_px.data',samples)
    np.savetxt('training/responses_px.data',nums)

def main():
    trainImages(itl_char.dictionary)

if __name__ == "__main__":
    main()

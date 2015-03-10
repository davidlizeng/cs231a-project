# Train character data

import cv2

import itl_char
import itl_paragraph

PATH = 'training/'

DICTIONARY_FILE = PATH + 'dictionary.txt'

# Name training images as 'train#.png', where # is the image number.
# Corresponding mapping file is 'map#.txt'. Map file should just be one line.
# Follow examples to add more training images.
NUM_IMAGES = 1

def readMapFile(filename):
    infile = open(filename, 'r')
    line = infile.readline()
    data = line.split()
    data = map(int, data)
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
    # Standardize character height/width
    # Although, if they come from itl_word and itl_line, height will be 20px
    # So, just shrink width alone
    # Follow StackOverflow post for pixel features

def trainImages(dictionary):
    for i in xrange(1, NUM_IMAGES + 1):
        trainImage(i)
    # Save results of all training into samples and responses data file
    # itl_char will be loading this

def main():
    trainImages(itl_char.dictionary)

if __name__ == "__main__":
    main()

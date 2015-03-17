import cv
import cv2
import numpy as np

DEST_FILE = 'train.png'
DEST_MAP_NAME = 'map'
[IMAGE_NAME, EXTENSION] = DEST_FILE.split('.')
imgnum = 2

def genRotateImage(img, r, center):
    rotMat = cv2.getRotationMatrix2D(center, r, 1)
    img_invert = (255 - img)
    img_r = cv2.warpAffine(img_invert, rotMat, (img.shape[1], img.shape[0]))
    img_r = (255 - img_r)
    return img_r

def genMaps(infile, outfile):
    with open(infile) as f:
        with open(outfile, "w") as f1:
            for line in f:
                f1.write(line) 

def genTransforms(img, map_file):
    global imgnum
    center = (img.shape[0]/2, img.shape[1]/2)
    cv2.imwrite(IMAGE_NAME + str(imgnum) + '.' + EXTENSION, img)
    genMaps(map_file, DEST_MAP_NAME + str(imgnum) + '.txt')
    imgnum += 1
    """
    for r in [-1,1]:
        img_r = genRotateImage(img,r,center)
        cv2.imwrite(IMAGE_NAME + str(imgnum) + '.' + EXTENSION, img_r)
        genMaps(map_file, DEST_MAP_NAME + str(imgnum) + '.txt')
        imgnum += 1
    for kx in range(1,2):
        for ky in range(1,2):
            for sx in range(1,2):
                for sy in range(1,2):
                    img_gb = cv2.GaussianBlur(img, (kx*2+1,ky*2+1), sx, sy)
                    cv2.imwrite(IMAGE_NAME + str(imgnum) + '.' + EXTENSION, img_gb)
                    genMaps(map_file, DEST_MAP_NAME + str(imgnum) + '.txt')
                    imgnum += 1
    for sy in range(1,3):
        scaleFactor = (1+(0.1*sy))
        img_s = cv2.resize(img, (0,0), fx=1, fy=scaleFactor)
        cv2.imwrite(IMAGE_NAME+str(imgnum) + '.' + EXTENSION, img_s)
        genMaps(map_file, DEST_MAP_NAME + str(imgnum) + '.txt')
        imgnum += 1
    for sx in range(1,3):
        scaleFactor = (1+(0.1*sx))
        img_s = cv2.resize(img, (0,0), fx=scaleFactor, fy=1)
        cv2.imwrite(IMAGE_NAME+str(imgnum) + '.' + EXTENSION, img_s)
        genMaps(map_file, DEST_MAP_NAME + str(imgnum) + '.txt')
        imgnum += 1
    """
    
MAP_FILES = ['map_ambiguous.txt', 'map_digit.txt', 'map_letter.txt', 'map_math_letter.txt', 'map_math_symbol.txt']    
IMAGE_FILES = ['train_ambiguous.png', 'train_digit.png', 'train_letter.png', 'train_math_letter.png', 'train_math_symbol.png']
TRAIN_DIR = 'train/'
LETTERS_DIR = 'letter/'
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

for i in range(len(MAP_FILES)):
    map_file = TRAIN_DIR +MAP_FILES[i]
    img = cv2.imread(TRAIN_DIR+IMAGE_FILES[i])
    genTransforms(img, map_file)

"""for i in range(len(LETTERS)):
    map_file = LETTERS_DIR+LETTERS[i]+'.txt'
    img = cv2.imread(LETTERS_DIR+LETTERS[i]+'.png')
    genTransforms(img, map_file)"""
        
        
        

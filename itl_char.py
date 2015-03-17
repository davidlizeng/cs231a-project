
import cv
import cv2
import numpy as np
from sklearn import linear_model

PATH = 'training/'
DICTIONARY_FILE = PATH + 'dictionary.txt'
SAMPLES_FILE = PATH + 'samples_fs.data'
RESPONSES_FILE = PATH + 'responses_fs.data'

REGULAR = 0
MATH = 1
AMBIGUOUS = 2
COUNTS = [0] * 3

CHAR_BOX = (20, 20)

dictionary = {}

# model = cv2.KNearest()  # KNearest
model = linear_model.LogisticRegression(C=1e5)  # LogReg
# model = cv2.SVM()

fs_max = None
fs_min = None
fs_steps = None
FS_STEP = 15

if len(dictionary) == 0:
    print 'Loading dictionary...'
    infile = open(DICTIONARY_FILE, 'r')
    infile.readline()
    mode = REGULAR
    for line in infile:
        if line.startswith('# Math'):
            mode = MATH
            continue
        elif line.startswith('# Ambiguous'):
            mode = AMBIGUOUS
            continue
        data = line.split()
        num = int(data[0])
        tex = (data[1], mode)
        dictionary[num] = tex
        COUNTS[mode] += 1
    infile.close()
    print 'Loading model...'
    samples = np.loadtxt(SAMPLES_FILE, np.float32)
    responses = np.loadtxt(RESPONSES_FILE, np.float32)
    print samples.shape, responses.shape
    fs_max = np.loadtxt(PATH + 'fs_max.data', np.float32)
    fs_min = np.loadtxt(PATH + 'fs_min.data', np.float32)
    fs_steps = (fs_max - fs_min) / FS_STEP
    responses = responses.reshape((responses.size, 1))
    # KNearest, SVM
    # model.train(samples, responses)
    # LogReg
    model.fit(samples, responses.ravel())


def getBinaryImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    unused, img = cv2.threshold(img, 200, 1, cv.CV_THRESH_BINARY_INV)
    return img

def parseCharacter(img, getScore=False):
    # cv2.imshow('IMG', img)
    # cv2.waitKey(0)

    # Frey-Slate attributes
    img = getFreySlateAttributes(img);

    # Pixel Values as features
    # img = cv2.resize(img, CHAR_BOX)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = np.float32(img.reshape((1, 400)))
    # img = img / np.linalg.norm(img)
    # print img.shape

    # KNearest
    # val, results, n_response, distances = model.find_nearest(img, k = 1)
    # score = distances[0][0]
    # SVM
    # val = model.predict(img)
    # score = 0
    # Log Reg
    val = model.predict(img)
    [index] = np.where(model.classes_ == val)
    score = model.predict_proba(img)[0][index][0]

    # print dictionary[int(val)], score

    if not getScore:
        return dictionary[int(val)]
    else:
        return dictionary[int(val)], score

def normalizeFreySlateAttributes(sample):
    sample = sample - fs_min
    sample = np.divide(sample, fs_steps)
    return sample

def getFreySlateAttributes(img):
    sample = np.zeros((10, 1))
    img = getBinaryImage(img)
    # The horizontal position, counting pixels from the left edge of the image, of the center
    # of the smallest rectangular box that can be drawn with all "on" pixels inside the box.
    columnSums = np.sum(img, axis=0)
    for i in xrange(0, len(columnSums)):
        if columnSums[i] != 0:
            sample[0] = i
            break
    # The vertical position, counting pixels from the bottom, of the above box.
    rowSums = np.sum(img, axis=1)
    for i in xrange(0, len(rowSums)):
        if rowSums[len(rowSums) - i - 1] != 0:
            sample[1] = i
            break
    # The width, in pixels, of the box.
    height = img.shape[0]
    width = img.shape[1]
    sample[2] = width
    # The total number of "on" pixels in the character image.
    totalOn = np.sum(columnSums)
    sample[3] = np.sum(columnSums)
    # The mean horizontal position of all "on" pixels relative to the center of the box and
    # divided by the width of the box. This feature has a negative value if the image is "leftheavy"
    # as would be the case for the letter L.
    centerCol = int(width / 2) if (width % 2 == 1) else int(width / 2) - 0.5
    total = 0
    for i in xrange(0, width):
        total += (i - centerCol) * columnSums[i] / width
    total = float(total) / totalOn
    sample[4] = total
    # The mean vertical position of all "on" pixels relative to the center of the box and divided
    # by the height of the box
    centerRow = int(height / 2) if (height % 2 == 1) else int(height / 2) - 0.5
    total = 0
    for i in xrange(0, height):
        total += (centerRow - i) * rowSums[i] / height
    total = float(total) / totalOn
    sample[5] = total
    # The mean squared value of the horizontal pixel distances as measured in 6 above. This
    # attribute will have a higher value for images whose pixels are more widely separated
    # in the horizontal direction as would be the case for the letters W or M.
    total = 0
    for i in xrange(0, width):
        total += (i - centerCol) * (i - centerCol) * columnSums[i]
    total = float(total) / totalOn
    sample[6] = total
    # The mean squared value of the vertical pixel distances as measured in 7 above.
    total = 0
    for i in xrange(0, height):
        total += (centerRow - i) * (centerRow - i) * rowSums[i]
    total = float(total) / totalOn
    sample[7] = total
    # The mean value of the squared horizontal distance times the vertical distance for each
    # "on" pixel. This measures the correlation of the horizontal variance with the vertical
    # position.
    total = 0
    for i in xrange(0, height):
        for j in xrange(0, width):
            if img[i][j] == 0: continue
            total += (j - centerCol) * (j - centerCol) * (centerRow - i)
    total = float(total) / totalOn
    sample[8] = total
    # The mean value of the squared vertical distance times the horizontal distance for each
    # "on" pixel. This measures the correlation of the vertical variance with the horizontal
    # position.
    total = 0
    for i in xrange(0, height):
        for j in xrange(0, width):
            if img[i][j] == 0: continue
            total += (centerRow - i) * (centerRow - i) * (j - centerCol)
    total = float(total) / totalOn
    sample[9] = total
    sample = normalizeFreySlateAttributes(np.transpose(sample))
    return np.float32(sample)

def main():
    pass

if __name__ == "__main__":
    main()


import cv2
import numpy as np
from sklearn import linear_model

PATH = 'training/'
DICTIONARY_FILE = PATH + 'dictionary.txt'
SAMPLES_FILE = PATH + 'samples_px.data'
RESPONSES_FILE = PATH + 'responses_px.data'

REGULAR = 0
MATH = 1
AMBIGUOUS = 2
COUNTS = [0] * 3

CHAR_BOX = (20, 20)

dictionary = {}

# model = cv2.KNearest()  # KNearest
model = linear_model.LogisticRegression(C=1e5)  # LogReg

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
    responses = responses.reshape((responses.size, 1))
    # KNearest, SVM
    # model.train(samples, responses)
    # LogReg
    model.fit(samples, responses.ravel())


def parseCharacter(img):
    img = cv2.resize(img, CHAR_BOX)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.float32(img.reshape((1, 400)))
    img = img / np.linalg.norm(img)

    # KNearest
    # val, results, n_response, distances = model.find_nearest(img, k = 1)
    # print dictionary[int(val)]
    # Log Reg
    val = model.predict(img)
    [index] = np.where(model.classes_ == val)
    print dictionary[int(val)], model.predict_proba(img)[0][index]

    return dictionary[int(val)]

def main():
    pass

if __name__ == "__main__":
    main()

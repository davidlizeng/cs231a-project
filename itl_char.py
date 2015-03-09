

PATH = 'training/'
DICTIONARY_FILE = PATH + 'dictionary.txt'

REGULAR = 0
MATH = 1
AMBIGUOUS = 2
COUNTS = [0] * 3

dictionary = {}
if len(dictionary) == 0:
    print 'Loading dictionary...',
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
    print 'Done'
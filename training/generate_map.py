# Toy file to generate map file

outfile = open('map1.txt', 'w')

text = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ\
        abcdefghijklmnopqrstuvwxyz\
        AaBbCcDdEeFfGgHhIiJjKkLlMm\
        NnOoPpQqRrSsTtUuVvWwXxYyZz'

for i in xrange(26):
  outfile.write('%d ' % (i + 26))
for i in xrange(26):
  outfile.write('%d ' % i)
for i in xrange(26):
  outfile.write('%d %d ' % (i + 26, i))

outfile.close()

## Dictionary numbering has been set up to allow for appending new entries
## Mapping files will become invalidated otherwise

DICTIONARY_FILE = 'dictionary.txt'

outfile = open(DICTIONARY_FILE, 'w')

num = 0

def output(s):
  global num
  outfile.write('%d %s\n' % (num, s))
  num += 1

outfile.write('# Regular\n')

# Lowercase and uppercase letters
for i in xrange(26):
  ch = chr(ord('a') + i)
  output(ch)
for i in xrange(26):
  ch = chr(ord('A') + i)
  output(ch)

outfile.write('# Math\n')
num = 1000

# Lowercase and uppercase letters
for i in xrange(26):
  ch = chr(ord('a') + i)
  output(ch)
for i in xrange(26):
  ch = chr(ord('A') + i)
  output(ch)

# Digits
for i in xrange(1, 11):
  output(str(i))

symbols = [
  '\\alpha',
  '\\beta',
  '\\delta',
  '\\gamma',
  '+',
  '-',
  '*',
  '>',
  '<',
  '\\leq',
  '\\geq',
  '=',
  '\\neq',
  '\\phi',
  '\\psi',
  '\\sigma'
]
for s in symbols:
  output(s)

outfile.write('# Ambiguous\n')
num = 2000

# Random punctuation
chars = '()[]{}.,/;:'
for ch in chars:
  output(ch)

outfile.close()

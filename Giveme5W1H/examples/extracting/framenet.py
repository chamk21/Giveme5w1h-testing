from pprint import pprint
from operator import itemgetter
from nltk.corpus import framenet as fn
from nltk.corpus.reader.framenet import PrettyList
f = fn.frame(424)
lex = f.lexUnit
for each in lex:
    lex_word = each[:-2]
